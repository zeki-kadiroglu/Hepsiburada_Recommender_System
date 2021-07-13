# Hepsiburada_Recommender_System
- ATTENTION
- pip install gensim ==3.8.3 NOT 4.0



# In Python Virtual Environment
  To run the docker file;

-  docker build -t CONTAINERNAME .
-  docker run -it -d -p 80:81 CONTAINERNAME
- P.S. you can adjust your port name in dockerfile and in your Virtual Machine.
- Now it is working in the AWS EC2 ubuntu free tier.

# POSTMAN REQUESTS
- POSTMAN documentation link ------------>> https://www.getpostman.com/collections/7a0ed085bf1d71f9ff5b
- GET request  list your favourite products ------>>  http://18.188.12.93/pick_product
- POST request click your product then show the recommendations ------->>> http://18.188.12.93/recommendation/Domates Salkım 500 gr
- Input should be "Domates Salkım 500 gr" after that link http://18.188.12.93/recommendation/
