import ijson
import multiprocessing
from multiprocessing import Pool
import numpy as np
from skimage.filters import threshold_otsu
from PIL import Image

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

import os
import boto3
from concurrent.futures import ThreadPoolExecutor
import io


### Function to load big json file 
def load_large_json(file_path):
    with open(file_path, 'r') as file:
        parser = ijson.parse(file)
        data = []
        item = {}
        key = None
        for prefix, event, value in parser:
            if event == 'start_map':
                item = {}
            elif event == 'end_map':
                data.append(item)
            elif event == 'map_key':
                key = value
                if key not in item:
                    item[key] = []
            elif event in ['string', 'number', 'boolean', 'null']:
                item[key].append(value)
    return data


### Function to process filtering process by multiprocessing
class ItemProcessor:
    def __init__(self, necessary_keywords):
        self.lang_keywords = ['Afrikaans', 'Arabic', 'Catalan/Valencian', 'Chinese', 'Czech', 'Danish', 'Dutch', 'English', 'Finnish', \
                             'French', 'German', 'Hawaiian', 'Hebrew', 'Hindi', 'Hungarian', 'Indonesian', 'Italian', 'Japanese', 'Korean', \
                             'Norwegian', 'Portuguese', 'Russian', 'Spanish', 'Swedish', 'Tagalog', 'Turkish', 'Ukrainian', 'Urdu', 'Welsh']
        self.other_keywords = []
        self.necessary_keywords = necessary_keywords

    def check_filters_keywords(self, item):
        if 'filters' in item:
            item_filters = set(item['filters'])
            if all(keyword in item_filters for keyword in self.necessary_keywords) and \
                all(keyword not in item_filters for keyword in self.lang_keywords) and \
                all(keyword not in item_filters for keyword in self.other_keywords):
                return item
        return None

    def process_item(self, item):
        return self.check_filters_keywords(item)

    def process_items(self, data):
        pool = multiprocessing.Pool()
        matching_items = pool.map(self.process_item, data)
        pool.close()
        pool.join()
        matching_items = [item for item in matching_items if item is not None]
        return matching_items


### Silhouette
def convert_image(image):
    if image.mode == 'RGBA':
        # Create a new RGBA image with white background
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

        # Convert RGBA image to RGB
        image = image.convert('RGB')
    return image

def resize_image(image, size):
    # Resize the image to the desired size
    image = image.resize(size)
    return image

def convert_image_to_bw(image_gray):   
    # Convert image to numpy array
    image_array = np.array(image_gray)
    
    # Calculate optimal threshold using Otsu's method
    threshold = threshold_otsu(image_array)
    
    # Apply binary threshold
    image_bw = image_gray.point(lambda x: 0 if x < threshold else 255, mode='L')
    
    return image_bw


### Count the number of nouns, adjectives and adverbs for "name" key and calculate the sum of them. Soft dict list in descending order by the sum. Implement this by multiprocessing
# Function to count the number of nouns, adjectives, and adverbs in a given sentence
class NounAdjectiveAdverbCounter:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

    def count_nouns_adjectives_adverbs(self, sentence):        
        nouns, adjectives, adverbs = 0, 0, 0
        tokens = word_tokenize(sentence.lower())
        tagged_tokens = pos_tag(tokens)

        for word, tag in tagged_tokens:
            if tag.startswith("N"):
                nouns += 1
            elif tag.startswith("J"):
                adjectives += 1
            elif tag.startswith("R"):
                adverbs += 1

        return nouns, adjectives, adverbs

    def compute_sum(self, item):
        name = item["name"][0]        
        nouns, adjectives, adverbs = self.count_nouns_adjectives_adverbs(name)
        total_sum = nouns + adjectives + adverbs

        # Add the new key "sum" with the calculated sum to the dictionary
        item_with_sum = item.copy()
        item_with_sum["sum"] = total_sum

        return item_with_sum

    def compute_sum_and_sort_by_sum(self, dict_list):
        # Use multiprocessing Pool to parallelize the computations
        with Pool() as pool:
            result = pool.map(self.compute_sum, dict_list)

        # Sort the list of dictionaries in descending order based on the sum
        sorted_result = sorted(result, key=lambda x: x["sum"], reverse=True)
        
        return sorted_result


### Choose best images from current large training cricut image dataset based on image and prompts similarity.
class DuplicateRemover:
    def __init__(self, data):
        self.data = data
        self.processed_data = None
        self.ps = PorterStemmer()

    def process_name(self, name):
        return {self.ps.stem(w) for w in word_tokenize(name)}

    def is_similar(self, processed_name1, processed_name2):
        return float(len(processed_name1 & processed_name2)) / float(max(len(processed_name1), len(processed_name2))) > 0.5

    def remove_duplicates(self, n = 0):
        self.processed_data = [(i, self.process_name(d['name'][0])) for i, d in enumerate(self.data)]
        i = 0
        while i < len(self.processed_data):
            current_name = self.processed_data[i][1]
            j = i + 1
            while j < len(self.processed_data):
                if self.is_similar(current_name, self.processed_data[j][1]):
                    del self.processed_data[j]
                else:
                    j += 1
            i += 1

        unique_data = [self.data[i] for i, _ in self.processed_data]
        
        # Remove the names with less than n words
        unique_data = [d for d in unique_data if len(d['name'][0].split()) >= n]

        # Sort the data by the number of words in the name in descending order
        unique_data.sort(key=lambda d: len(d['name'][0].split()), reverse=True)

        return unique_data
    

### Upload images to S3
class ImageUploader:
    def __init__(self, bucket_name, subfolder, image_type, size):
        self.bucket_name = bucket_name
        self.subfolder = subfolder
        self.image_type = image_type
        self.size = size
        self.s3_client = boto3.client('s3')
        self.s3_client.put_object(Bucket=self.bucket_name, Key=f"{self.subfolder}/")

    def convert_image(self, image):
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
            image = image.convert('RGB')
        return image

    def resize_image(self, image):
        image = image.resize(self.size)
        return image

    def upload_image(self, item, index):
        url = item['previewUrl'][0]
        name = item['name'][0]
        response = requests.get(url)
        image_content = response.content
        image = Image.open(io.BytesIO(image_content))
        image = self.convert_image(image)
        image = self.resize_image(image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        #image_name = str(index)
        image_name = str(item['id'][0])
        image_key = f"{self.subfolder}/{image_name}.png"
        self.s3_client.put_object(Bucket=self.bucket_name, Key=image_key, Body=image_bytes)
        text_content = self.image_type + ", " + name.lower()
        text_key = f"{self.subfolder}/{image_name}.txt"
        self.s3_client.put_object(Bucket=self.bucket_name, Key=text_key, Body=text_content)

    def upload_images_to_s3(self, items):
        self.upload_image(items[0], 0)
        with ThreadPoolExecutor() as executor:
            executor.map(self.upload_image, items, range(len(items)))
            

### Upload images to S3 after converting to BW image
class BWImageUploader:
    def __init__(self, items, bucket_name, subfolder, image_type, size):
        self.items = items
        self.bucket_name = bucket_name
        self.subfolder = subfolder
        self.image_type = image_type
        self.size = size
        self.s3_client = boto3.client('s3')
        self.s3_client.put_object(Bucket=self.bucket_name, Key=f"{self.subfolder}/")

    def convert_image(self, image):
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
            image = image.convert('RGB')
        return image

    def resize_image(self, image):
        return image.resize(self.size)

    def convert_image_to_bw(self, image_gray):
        image_array = np.array(image_gray)
        threshold = threshold_otsu(image_array)
        image_bw = image_gray.point(lambda x: 0 if x < threshold else 255, mode='L')
        return image_bw

    def upload_image(self, item, index):
        url = item['previewUrl'][0]
        name = item['name'][0]
        response = requests.get(url)
        image_content = response.content
        image = Image.open(io.BytesIO(image_content))
        image = self.convert_image(image)
        image = self.resize_image(image)
        image = image.convert('L') 
        image_binary = self.convert_image_to_bw(image)
        image_bytes = io.BytesIO()
        image_binary.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        #image_name = str(index)
        image_name = str(item['id'][0])
        image_key = f"{self.subfolder}/{image_name}.png"
        self.s3_client.put_object(Bucket=self.bucket_name, Key=image_key, Body=image_bytes)
        text_content = self.image_type + ", " + name.lower()
        text_key = f"{self.subfolder}/{image_name}.txt"
        self.s3_client.put_object(Bucket=self.bucket_name, Key=text_key, Body=text_content)

    def upload_images_bw_to_s3(self):    
        self.upload_image(self.items[0], 0)
        with ThreadPoolExecutor() as executor:
            executor.map(self.upload_image, self.items, range(len(self.items)))
            
            
class S3FileProcessor:
    def __init__(self, s3_bucket, s3_client=None):
        self.s3_bucket = s3_bucket
        self.s3_client = s3_client or boto3.client('s3')

    def download_files(self, subfolder, temp_folder):
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        
        downloaded_files = []
        continuation_token = None

        while True:
            list_objects_params = {
                'Bucket': self.s3_bucket,
                'Prefix': subfolder,
                'MaxKeys': 1000,
            }

            if continuation_token:
                list_objects_params['ContinuationToken'] = continuation_token

            s3_objects = self.s3_client.list_objects_v2(**list_objects_params)
            if 'Contents' in s3_objects:
                for obj in s3_objects['Contents']:
                    file_key = obj['Key']
                    file_name = os.path.basename(file_key)
                    if file_name == "":
                        continue
                    local_path = os.path.join(temp_folder, file_name)
                    self.s3_client.download_file(self.s3_bucket, file_key, local_path)
                    downloaded_files.append(local_path)
            
            if 'NextContinuationToken' in s3_objects:
                continuation_token = s3_objects['NextContinuationToken']
            else:
                break

        return downloaded_files

    def get_file_extension(self, file_path):
        _, file_extension = os.path.splitext(file_path)
        return file_extension

    
    def convert_image(self, image):
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
            image = image.convert('RGB')
        return image

    def convert_image_to_bw(self, image_gray):
        image_array = np.array(image_gray)
        threshold = threshold_otsu(image_array)
        image_bw = image_gray.point(lambda x: 0 if x < threshold else 255, mode='L')
        return image_bw
    
    def process_files(self, files, temp_folder, mode):
        processed_files = []
        print(len(files))
        for file in files:            
            if self.get_file_extension(file) == '.png':
                original_image = Image.open(file)
                if mode == 1:                    
                    image = self.convert_image(original_image)
                else:
                    image = self.convert_image(original_image)
                    image = image.convert('L') 
                    image = self.convert_image_to_bw(image)
                #processed_file = f'processed_{os.path.basename(file)}'
                #processed_path = os.path.join(temp_folder, processed_file)
                image.save(file)
            processed_files.append(file)
        return processed_files

    def upload_files(self, subfolder, files):
        for file in files:
            s3_key = os.path.join(subfolder, os.path.basename(file))
            self.s3_client.upload_file(file, self.s3_bucket, s3_key)

    def delete_temp_folder(self, temp_folder):
        for file in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_folder)

    def process_and_upload(self, input_subfolder, output_subfolder, temp_folder, mode):
        downloaded_files = self.download_files(input_subfolder, temp_folder)
        processed_files = self.process_files(downloaded_files, temp_folder, mode)
        self.upload_files(output_subfolder, processed_files)
        self.delete_temp_folder(temp_folder)