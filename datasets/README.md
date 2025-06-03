## Important nots
After downloading the Recipe1M dataset, you will see two files named `layer1.json` and `layer2.json`.

Then we format it into two .tsv files (`recipe1m_train.tsv` and `recipe1m_val.tsv`) as follows by running the `recipe1m_processing.py`:
```
caption image
This is food title: Kombu Tea Grilled Chicken Thigh This is ingredient: ['2 chicken thighs', '2 tsp kombu tea', '1 white pepper'] This is instruction: ['pierce the skin of the chicken with a fork or knife.', 'sprinkle with kombu tea evenly on both sides of the chicken, about 1 teaspoon per chicken thigh.', 'brown the skin side of the chicken first over high heat until golden brown.', 'sprinkle some pepper on the meat just before flipping over.', 'then brown the other side until golden brown.']      6bdca6e490.jpg
This is food title: Strawberry Rhubarb Dump Cake This is ingredient: ['6 8 cups fresh rhubarb, or', '6 8 cups frozen rhubarb, thawed', '1 12 cups granulated sugar', '6 ounces strawberry jell o gelatin dessert', '1 white cake mix', '1 12 cups water', '12 cup butter or 12 cup margarine, melted'] This is instruction: ['put ingredients in a buttered 9 x 12 x 2 inch pan in even layers in the order that they are given do not mix.', 'bake in a 350 oven for 1 hour.'] 6409eab844.jpg
This is food title: Yogurt Parfaits This is ingredient: ['8 ounces, weight light fat free vanilla yogurt', '1 cup fresh sliced strawberries', '1/4 cups low fat granola'] This is instruction: ['layer all ingredients in a serving dish.']     a1374cdd98.jpg
```
where each line contains the caption followed by the filename of the image files. Save these `.tsv` files into the `dataset/` folder (the default names expected are `recipe1m_train.tsv` and `recipe1m_val.tsv`). The repo contains two placeholder files with a few examples, and you will have to replace them with the appropriate data.

