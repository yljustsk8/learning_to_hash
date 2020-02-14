# Learning to Hash for Efficient Search over Incomplete Knowledge Graphs

- Intorduction

    - The code is a first implementation of our [paper](https://ieeexplore.ieee.org/abstract/document/8970688) at ICDM 2019, including the train process of BIO data;
    - Our code is based on [GraphQEmbed](https://github.com/williamleif/graphqembed), you can get the original code and BIO data on that page.
    
- Requirements
    
    - see `./netquery/requirements.txt`.
    
- Running the code

    - Download the data and unzip it at `./` and reach `./netquery/bio/`;
    
    - We transfer vector x to tanh(δx) and gradually increase δ(from 1 to b) in order to bring x closer to the hamming space. In the code, we used parameter "beta" to represent δ and b is set to 20 in our experiment.;
    
    - Run `python train.py --lr 0.001 --beta β --pretrain True` to use the edge data to train the embedding, where β gradually increased from 1 to b-1;
    
    - Then run `python train.py --lr 0.001 --beta b --pretrain False` to use the query data to fine-tune the embedding;
    
    - Run `python test.py --beta b` to compare the difference between the original and the hashed model. 

- Other information

    - The code is currently maintained by Yinlin Jiang. You can contact me at [yljiang@seu.edu.cn](mailto:yljiang@seu.edu.cn);
    - This code is a part of our full experiment, we are working on to release a more complete version.  