# import PIL.Image
# import google.generativeai as genai
# from PIL import Image, ImageDraw
# import json
# import io

# # Configure Gemini Vision API
# genai.configure(api_key='AIzaSyAitCY6WKt5IvcVbuTlQjG00ntdzXRZ82M')
# model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')


# # Input image path
# image_path = "C:/Users/athar/OneDrive/Desktop/Project/edi-flow/img/img7.jpg"

# # Example polygon data: can be nested
# polygon_list = [
#     [(731, 313), (727, 67), (916, 62), (920, 311), (867, 311)],
#     [(930, 590), (840, 497), (718, 574), (813, 581)],
#     [(837, 916), (894, 772), (903, 772), (751, 775)]
# ]


# def crop_polygon_region(image, polygon):
#     """Crop a polygonal region from an image."""
#     clean_polygon = []
#     for point in polygon:
#         if isinstance(point, (list, tuple)) and len(point) == 2:
#             clean_polygon.append((int(point[0]), int(point[1])))
#         else:
#             raise ValueError(f"Invalid point: {point}")

#     mask = Image.new('L', image.size, 0)
#     ImageDraw.Draw(mask).polygon(clean_polygon, outline=1, fill=255)
#     result = Image.new('RGB', image.size)
#     result.paste(image, mask=mask)
#     bbox = mask.getbbox()
#     cropped = result.crop(bbox)
#     return cropped

# def analyze_shape_text(label, shape_img):
#     """Use Gemini Vision to identify the text inside the shape."""
#     prompt = """You are given a cropped image of a shape in a hand-drawn diagram.

# Your job is to extract ONLY the text written inside the shape.

# Respond with ONLY the plain text content, without any explanation or formatting."""

#     # Convert image to in-memory bytes
#     img_byte_arr = io.BytesIO()
#     shape_img.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)

#     # Send prompt and image to Gemini
#     response = model.generate_content([prompt, PIL.Image.open(img_byte_arr)], stream=False)
#     return label, response.text.strip()


#     # Convert image to in-memory bytes
#     img_byte_arr = io.BytesIO()
#     shape_img.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)

#     # Send prompt and image to Gemini
#     response = model.generate_content([prompt, PIL.Image.open(img_byte_arr)], stream=False)
#     return label, response.text

# def process_image_polygons(image_path, polygons):
#     """Iterate through all polygons, crop, analyze, and collect text results as a list."""
#     main_img = Image.open(image_path)
#     results = []

#     for polygon in polygons:
#         try:
#             cropped = crop_polygon_region(main_img, polygon)
#             _, text_only = analyze_shape_text("", cropped)
#             results.append(text_only)
#         except Exception as e:
#             results.append(f"error: {e}")

#     return results


# # Run the full process and print results
# output = process_image_polygons(image_path, polygon_list)
# print(json.dumps(output, indent=2))









import PIL.Image
import google.generativeai as genai
from PIL import Image, ImageDraw
import io


genai.configure(api_key='AIzaSyAitCY6WKt5IvcVbuTlQjG00ntdzXRZ82M')
model = genai.GenerativeModel(model_name='models/gemini-1.5-flash')

def extract_texts_from_polygons(image_path, polygon_list):
    """
    Given an image path and a list of polygons (each polygon is a list of (x,y) tuples),
    crop each polygonal region from the image,
    send it to Gemini Vision to extract only the text inside,
    and return a list of text strings.
    """

    def crop_polygon_region(image, polygon):
        clean_polygon = []
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                clean_polygon.append((int(point[0]), int(point[1])))
            else:
                raise ValueError(f"Invalid point: {point}")

        mask = Image.new('L', image.size, 0)
        ImageDraw.Draw(mask).polygon(clean_polygon, outline=1, fill=255)
        result = Image.new('RGB', image.size)
        result.paste(image, mask=mask)
        bbox = mask.getbbox()
        cropped = result.crop(bbox)
        return cropped

    def analyze_shape_text(shape_img):
        prompt = """You are given a cropped image of a shape in a hand-drawn diagram.

Your job is to extract ONLY the text written inside the shape.

Respond with ONLY the plain text content, without any explanation or formatting."""

        img_byte_arr = io.BytesIO()
        shape_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        response = model.generate_content([prompt, PIL.Image.open(img_byte_arr)], stream=False)
        return response.text.strip()

    main_img = Image.open(image_path)
    results = []

    for polygon in polygon_list:
        try:
            cropped = crop_polygon_region(main_img, polygon)
            text = analyze_shape_text(cropped)
            results.append(text)
        except Exception as e:
            results.append(f"error: {e}")

    return results




polygons = [
    [(731, 313), (727, 67), (916, 62), (920, 311), (867, 311)],
    [(930, 590), (840, 497), (718, 574), (813, 581)],
    [(837, 916), (894, 772), (903, 772), (751, 775)]
]

texts = extract_texts_from_polygons("C:/Users/athar/OneDrive/Desktop/Project/edi-flow/img/img7.jpg", polygons)
print(texts) 

