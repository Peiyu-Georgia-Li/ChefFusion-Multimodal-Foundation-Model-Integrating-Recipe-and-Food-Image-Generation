# ChefFusion: Multimodal Foundation Model Integrating Recipe and Food Image Generation

## Abstract
Significant work has been conducted in the domain of food computing, yet these studies typically focus on single tasks such as t2t (instruction generation from food titles and ingredients), i2t (recipe generation from food images), or t2i (food image generation from recipes). None of these approaches integrate all modalities simultaneously.
To address this gap, we introduce a novel food computing foundation model that achieves true multimodality, encompassing tasks such as t2t, t2i, i2t, it2t, and t2ti. By leveraging large language models (LLMs) and pre-trained image encoder and decoder models, our model can perform a diverse array of food computing-related tasks, including food understanding, food recognition, recipe generation, and food image generation.
Compared to previous models, our foundation model demonstrates a significantly broader range of capabilities and exhibits superior performance, particularly in food image generation and recipe generation tasks.


## Case Study and the Architecture of ChefFusion
![case](https://github.com/user-attachments/assets/64556d24-45c2-44be-88de-27dcfa1ed53c "Case Study: ChefFusion demonstrates a wide suite of multimodal capabilities, including food understanding, food recognition, recipe generation, food image generation and multimodal dialogue (left). Example of food images generated by ChefFusion (right).")
**Case Study**: ChefFusion demonstrates a wide suite of multimodal capabilities, including food understanding, food recognition, recipe generation, food image generation and multimodal dialogue (left). Example of food images generated by ChefFusion (right).

![pipeline-1](https://github.com/user-attachments/assets/5910581a-f9c6-443b-b75c-663cca880a64 "The architecture of ChefFusion: (1) Left: training the model to generate recipe by minimizing $l_{r}(x, y)$; (2) Right: training the model to generate food images by minimizing $l_{g}(y)$ and determine whether to produce text or images at each step by minimizing $l_{p}(y)$.")
**The architecture of ChefFusion**: (1) Left: training the model to generate recipe by minimizing $l_{r}(x, y)$; (2) Right: training the model to generate food images by minimizing $l_{g}(y)$ and determine whether to produce text or images at each step by minimizing $l_{p}(y)$.


<p align="center">
<img alt="Inference procedure for ChefFusion: The model takes in image and text inputs, and generate text interleaved with food image." src="https://github.com/user-attachments/assets/0a6c0436-ccc6-4270-8449-50aa5bf7a40d " width=500/>
</p>

**Inference procedure for ChefFusion**: The model takes in image and text inputs, and generate text interleaved with food image.

## How to Set Up
### Environment
Set up a new virtualenv, and install required libraries:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add the `chefFusion` library to PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/home/path/to/chefFusion/
```
## Training

### Preparing the Dataset

Our model is trained on the [Recipe1M](https://github.com/torralba-lab/im2recipe?tab=readme-ov-file#recipe1m-dataset) dataset. After following the instructions on the website to download the dataset, format it into a `.tsv` file as follows:

```
caption image
This is food title: Kombu Tea Grilled Chicken Thigh This is ingredient: ['2 chicken thighs', '2 tsp kombu tea', '1 white pepper'] This is instruction: ['pierce the skin of the chicken with a fork or knife.', 'sprinkle with kombu tea evenly on both sides of the chicken, about 1 teaspoon per chicken thigh.', 'brown the skin side of the chicken first over high heat until golden brown.', 'sprinkle some pepper on the meat just before flipping over.', 'then brown the other side until golden brown.']      6bdca6e490.jpg
This is food title: Strawberry Rhubarb Dump Cake This is ingredient: ['6 8 cups fresh rhubarb, or', '6 8 cups frozen rhubarb, thawed', '1 12 cups granulated sugar', '6 ounces strawberry jell o gelatin dessert', '1 white cake mix', '1 12 cups water', '12 cup butter or 12 cup margarine, melted'] This is instruction: ['put ingredients in a buttered 9 x 12 x 2 inch pan in even layers in the order that they are given do not mix.', 'bake in a 350 oven for 1 hour.'] 6409eab844.jpg
This is food title: Yogurt Parfaits This is ingredient: ['8 ounces, weight light fat free vanilla yogurt', '1 cup fresh sliced strawberries', '1/4 cups low fat granola'] This is instruction: ['layer all ingredients in a serving dish.']     a1374cdd98.jpg
```
where each line contains the caption followed by the filename of the image files. Save these `.tsv` files into the `dataset/` folder (the default names expected are `cc3m_train.tsv` and `cc3m_val.tsv`). The repo contains two placeholder files with a few examples, and you will have to replace them with the appropriate data.

The corresponding image files should be saved in the `data/` directory. The directory can be changed with the `--image-dir` runtime flag.



