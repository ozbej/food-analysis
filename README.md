# AIR Food Analysis Project

## NER Pipeline for Extracting Ingredients From Recipe Directions

The goal of this task is to train a Named Entity Recognition (NER) pipeline
to extract ingredients from recipe directions. Input to the pipeline are recipe
directions, and output of the pipeline are the extracted ingredients from the
recipe.  
**Example input**: *Thoroughly cream shortening, sugar and vanilla. Beat in
eggs, then chocolate. Sift together dry ingredients. Blend in with milk; add nuts.
Chill 3 hours; form in 1-inch balls and roll in powdered sugar. Place on greased
cookie sheet 2 to 3 inches apart. Bake at 350 for 15 minutes. Cool slightly
and remove from pan. Makes 4 dozen.*  
**Example output**: *shortening, sugar, vanilla, eggs, chocolate, flour, baking
powder, salt, milk, nuts*

The code for this task is located in `/ingredient_NER/` folder, with the following notebooks:
- `ingredient_NER_tagging.ipynb`, which contains the code for tagging ingredients in IOB tagging format (O, B-ING, I-ING),
- `ingredient_NER_training.ipynb`, which contains the code for fine-tuning a pretrained BERT model for NER, and
- `ingredient_NER_test_infer.ipynb`, which contains the code for evaluating our results on an independent test set and for infering from a recipe in text form.

## Recipe Recommendation

Code can be found in the `/recipe_recommendation/data_analysis_and_tfidf_recommendations` folder: 

### Preprocessing and data analysis
    check_ingredient_list_to_description: Used to analyze the step by step description and the corresponding ingredient list of recipes.
    sort_tags_into_groups: Used to sort tags into label groups by hand, used in the multi target learning. 

### Recommendation based on another Recipe (TFIDFVectorizer):
    Pass the index of a recipe to get a list of similar recipes. 

### Recommendation based on ingredient list (TFIDFVectorizer):
    Pass ingredients to get matching recipes.


Code can be found in the `/recipe_recommendation/******` folder: 