A Transfer Learning application designed to classify waste as **Trash** or **Recycle** on edge devices without internet connectivity. A model is trained on sample data is then used in a simple script that takes an image as imput and determines if the image is to be discarded as trash or recycle. This output could be integrated into a functioning system that triggers the item to be discarded in the appropriate bin.

## Prerequisites
- Python 3.8+
- A dataset of images (See Setup)

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

Data sourced from https://www.kaggle.com/datasets/techsash/waste-classification-data

The idea of the this project stems from the statistic that 70%+ of american recycle materials are discarded as trash, the concept of this solution is to design an autosorting trash bin that is functional to allow for for of the recyclable materials to be sorted and discarded appropriately. The MVP product would consist of:
1. A two compartment trash bin
2. a motorized platform that can rotate on its center axis to direct the content to either the right bin (trash) or the left bin (recycle)
3. A camera to take images of discarded items
4. Classification application that can run on edge device to determine where an item needs to go

This repo provides a simple example of item #4 only
