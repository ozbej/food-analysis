import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle


def make_it_a_list(string_thing):
    return [thing.strip('\'\'') for thing in string_thing.strip('][').split(', ')]


def make_it_a_string(a_list):
    a_str = ""
    for thing in a_list:
        a_str += " " + thing
    return a_str


def tf_idf(recipe):
    tfidf = TfidfVectorizer()
    recipe = recipe.apply(lambda x: make_it_a_string(make_it_a_list(x)))
    return tfidf.fit_transform(recipe)


def get_similar_recipe_from_ingredients_of_recipe(recipe, index, recipes):
    rec_tfidf = tf_idf(recipe)
    rec_cos_sim = linear_kernel(rec_tfidf, rec_tfidf[index])
    print("Original: " + recipes["name"][index] + " " + recipe[index])
    for ind, sim in enumerate(rec_cos_sim):
        if sim[0] > 0.70:
            print(str(recipes["name"][ind]))


def get_recipe_from_ingredient_list(ingredient_string, recipes):
    recipe_ingredients = recipes["ingredients"]
    new = pd.concat([pd.Series([ingredient_string]), recipe_ingredients], ignore_index=True)

    rec_tfidf = tf_idf(new)
    rec_cos_sim = linear_kernel(rec_tfidf, rec_tfidf[0])

    print("Ingredient looked for: " + new[0])
    for ind, sim in enumerate(rec_cos_sim):
        if ind == 0:
            continue
        if sim[0] > 0.7:
            print(str(ind) + " " + str(recipes['name'][ind - 1]) + " " + str(new[ind]))


def preprocess_interaction_by_user():
    interactions = pd.read_csv("data/RAW_interactions.csv")

    interactions_by_user = {}
    for interaction in interactions.iterrows():
        interaction = interaction[1]
        if interaction['user_id'] in interactions_by_user.keys():
            interactions_by_user[interaction['user_id']].append({'recipe_id': interaction['recipe_id'], 'review': interaction['review'], 'rating': interaction['rating']})
        else:
            interactions_by_user[interaction['user_id']] = [{'recipe_id': interaction['recipe_id'], 'review': interaction['review'], 'rating': interaction['rating']}]

    interactions_by_user_5_interactions = {}
    for user in interactions_by_user:
        if len(interactions_by_user[user]) > 4:
            interactions_by_user_5_interactions[user] = interactions_by_user[user]

    return interactions_by_user_5_interactions


def get_ingredients_from_recipe_id(recipes, id):
    return make_it_a_list(recipes.loc[recipes['id'] == id]['ingredients'].values[0])


def get_ingredients_sorted(interactions_by_user, recipes):
    ingredients_list = {"Test"}

    for user in interactions_by_user:
        for interaction in interactions_by_user[user]:
            ingredients = get_ingredients_from_recipe_id(recipes, interaction['recipe_id'])
            for ing in ingredients:
                if ing not in ingredients_list:
                    ingredients_list.add(ing)

    print(ingredients_list)
    print(len(ingredients_list))


def check_ingredient_list_to_description(recipes):
    count = 0
    for rec in recipes.iterrows():
        print(rec[1]['ingredients'])
        print(rec[1]['steps'])
        count += 1
        if count > 10:
            break


def user_based_stuff(interactions_by_user, recipes):
    get_ingredients_sorted(interactions_by_user, recipes)


def sort_tags_into_groups(recipes):
    tags_count = {}
    for rec in recipes.iterrows():
        tags = make_it_a_list(rec[1]['tags'])
        for tag in tags:
            if tag in tags_count.keys():
                tags_count[tag] += 1
            else:
                tags_count[tag] = 1

    sorted_tags_count = dict(sorted(tags_count.items(), key=lambda item: item[1], reverse=True))

    count = 0
    for tag in sorted_tags_count:
        container_id = input(tag)
        diet_tags = open("groups/diet.txt", 'a')
        country_tags = open("groups/country.txt", 'a')
        time_tags = open("groups/time.txt", 'a')
        occasion_tags = open("groups/occasion.txt", 'a')
        dish_type_tags = open("groups/type.txt", 'a')

        if container_id is "0":
            diet_tags.write(tag)
            diet_tags.write('\n')
        elif container_id is "1":
            country_tags.write(tag)
            country_tags.write('\n')
        elif container_id is "2":
            time_tags.write(tag)
            time_tags.write('\n')
        elif container_id is "3":
            occasion_tags.write(tag)
            occasion_tags.write('\n')
        elif container_id is "6":
            dish_type_tags.write(tag)
            time_tags.write('\n')

        diet_tags.close()
        country_tags.close()
        time_tags.close()
        occasion_tags.close()
        dish_type_tags.close()

        count += 1
        if count == 200:
            break


######################################################
### Kaggle Dataset
### https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
######################################################

user_interactions_needed = False
create_files = False
if __name__ == '__main__':
    if user_interactions_needed:
        if create_files:
            interactions_by_user = preprocess_interaction_by_user()
            with open('data_created/user_dump.pickle', 'wb') as f:
                pickle.dump(interactions_by_user, f)
        else:
            with open('data_created/user_dump.pickle', 'rb') as f:
                interactions_by_user = pickle.load(f)

    recipes = pd.read_csv("data/RAW_recipes.csv")

    ### Checking data ###
    # check_ingredient_list_to_description(recipes)
    # sort_tags_into_groups(recipes)

    ### Similarity ###
    # get_similar_recipe_from_ingredients_of_recipe(recipes["ingredients"], 6666, recipes)
    # get_recipe_from_ingredient_list("['salt', 'vinegar', 'potatoes']", recipes)
