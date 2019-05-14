
# from flask import jsonify
import os
from flask import send_file
from flask_restplus import Resource, fields, Namespace
from optimizer import Optimizer
from read_image import ReadImage

api = Namespace('Style Transfer', description='Stylize an image')

@api.route('/styletransfer')
class TransformEndpoint(Resource):
    inputs = api.model(
    "styletransfer", 
    { 
        "content_url": fields.String,
        "style_url": fields.List(fields.String),
        "mod_aspect_ratio": fields.Float(default=1.0),
        "base_width": fields.Integer,
        "iterations": fields.Integer(default=500),
        "checkpoint_iter": fields.Integer(default=100),
        "content_weight": fields.Float(default=5.0),
        "style_weight": fields.Float(default=500.0),
        "style_image_influence": fields.List(fields.Float(default=1.0)),
        "eval_content_layers": fields.List(fields.String, default=["conv4_2", "conv5_2"]),
        "eval_style_layers": fields.List(fields.String, default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]),
        "content_layer_influence": fields.List(fields.Float),
        "style_layer_influence": fields.List(fields.Float),
        "learning_rate": fields.Float(default=1.0),
        "noise_ratio": fields.Float(defualt=0.2)
        })

    @api.expect(inputs)
    def post(self):
        print(api.payload)

        MODEL_PATH = "./model/imagenet-vgg-verydeep-19.mat"
        SAVE_PATH = "./output/"
        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        payload = api.payload
        content_image, style_images = ReadImage(payload["base_width"], payload["mod_aspect_ratio"]).read_from_url(payload["content_url"], payload["style_url"])
        # Instantiate the optimizer
        optimizer = Optimizer(iterations=payload["iterations"], 
                            checkpoint_iter=payload["checkpoint_iter"],
                            style_image_influence=payload["style_image_influence"], 
                            eval_content_layers=payload["eval_content_layers"],
                            eval_style_layers=payload["eval_style_layers"],
                            content_weight=payload["content_weight"], 
                            style_weight=payload["style_weight"],
                            noise_ratio=payload["noise_ratio"],
                            learning_rate=payload["learning_rate"],
                            content_image=content_image,
                            style_images=style_images,
                            model_path=MODEL_PATH,
                            save_path=SAVE_PATH,
                            content_layer_influence=payload["content_layer_influence"],
                            style_layer_influence=payload["style_layer_influence"]
                            )
    
        # Execute optimizer
        optimizer.execute()

        return send_file("./output/generated_image.jpg")
