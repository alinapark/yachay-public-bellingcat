# Project name: Yachay AI, Telegram Bot

## Team Members
[Yachay AI team](https://www.yachay.ai/)

## Tool Description
We see an untapped potential in the development of OSINT tools for text-based geolocation. Hours of researchers' work to find, clean and process images/texts to identify a person's location could have been reduced to seconds with automated text analysis. 

**As NLP developers, we decided to tackle this challenge by creating an environment:**
1. For investigators to communicate with the models in the simplest way possible 
2. For other developers to contribute to our project and help us improve the models

This project is a new take on the [models](https://github.com/1712n/yachay-public/tree/master/conf_geotagging_model) previously released by Yachay AI: we are swapping the existing infrastructure with a ByT5 model, and setting up a simple User Interface for people to see it in action.

Now, one can simply text our bot on telegram and receive an analysis - life doesn't get easier than that 😌😌😌

## Installation
**There is no need to install anything to use this project! Simply text our [telegram bot](https://t.me/YachayAi_Bot) at [@YachayAi_Bot](https://t.me/YachayAi_Bot)**

In case you want to create your own version of the tool, the data sets can be found in the Google Drive [here](https://drive.google.com/file/d/1thkE-hgT3sDtZqILZH17Hyayy0hkk_jh/view?usp=share_link) and [here](https://drive.google.com/drive/folders/1P2QUGFBKaqdpZ4xAHmJMe2I57I94MJyO?usp=sharing).


#### Dependencies:
- pytorch == 1.7.1
- numpy == 1.19.2
- scikit-learn == 0.22.2
- tqdm == 4.62.3
- pandas == 1.0.3

The new, improved infrastructure created for this hackathon:

- `ByT5_model.py` - model implementation
- `train_model.py` - training code
- `utils.py` - inference code

Heads Up: this project is a demo for the hackathon. More updates on the models will be in the [main repository](https://github.com/1712n/yachay-public/tree/master/conf_geotagging_model):)

## Usage
To use the bot, follow the [link](https://t.me/YachayAi_Bot), or go to telegram and add @YachayAi_Bot

<img width="736" alt="step" src="https://user-images.githubusercontent.com/29067628/233823060-0561fbbf-3197-4523-a996-74fee3768c21.png">

1. Click on "Start"

<img width="720" alt="s2" src="https://user-images.githubusercontent.com/29067628/233824316-f500b6ff-7dde-43a5-85e9-13f32139ca55.png">


2. The bot introduces itself:)

3. Now send a text - anything under 300 characters would work. Give the bot a couple of moments to reply

4. <img width="718" alt="step3" src="https://user-images.githubusercontent.com/29067628/233823034-7eea115d-d90f-42ea-aa09-93509b2a8de8.png">

5. Done ✅ - easy 😔😔😔


## Additional Information



The Telegram bot serves as a demo of the models' potential - it's a work-in-progress and exists for demonstration purposes only. 

Still, we are providing all the [data and infrastructure code](https://github.com/1712n/yachay-public/tree/master/conf_geotagging_model) to make sure other developers can contribute:)



### Contact 📩

If you would like to contact us with any questions, concerns, or feedback, help@yachay.ai is our email.

You also can check out our site, [yachay.ai](https://www.yachay.ai/), or any of our socials below.


### Social Media 📱


<a href="https://discord.gg/msWFtcfmwe"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670157.png" width=5% height=5%></img></a>     <a href="https://twitter.com/YachayAi"><img src="https://cdn-icons-png.flaticon.com/128/3670/3670151.png" width=5% height=5%></img></a>     <a href="https://www.reddit.com/user/yachay_ai"><img src="https://cdn-icons-png.flaticon.com/512/3670/3670226.png" width=5% height=5%></img></a>

