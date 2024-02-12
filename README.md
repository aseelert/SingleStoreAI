### All you need is AI
#### Optain an OpenAI key to generate embeddings
https://platform.openai.com/api-keys

#### Prepare you datastore using SingleStore Database
create a free tier account database at https://portal.singlestore.com

<img width="1097" alt="image" src="https://github.com/aseelert/SingleStoreAI/assets/34473004/7b7572ee-498d-499d-a813-483befb0fac9">

#### Create the Database structure to store the Instagram posts and comments
createTable.sql
<img width="1203" alt="image" src="https://github.com/aseelert/SingleStoreAI/assets/34473004/47e37b04-1c78-4258-988a-658d726f9937">

#### Store your credentials into vault using a Master Password
```
python passwords.py -create -type instagram
```
```
python passwords.py -create -type singlestore
```
```
python passwords.py -create -type openai
```

#### Fetch Posts and related comments and store it directly into Singlestore Vector
```
./instaFetchPosts.py --master-password masterpassword --topic bmw --posts 2 --comments 20
```

#### RAG: search vector data matching to the search string you can enter
```
./instaSemanticSearch.py --master-password masterpassword
```
<img width="709" alt="image" src="https://github.com/aseelert/SingleStoreAI/assets/34473004/9c30f4a5-854f-4570-baa8-4d424e469cc9">
<img width="1540" alt="image" src="https://github.com/aseelert/SingleStoreAI/assets/34473004/66c02305-487c-41cb-b13b-dc6a8a0d1590">
<img width="513" alt="image" src="https://github.com/aseelert/SingleStoreAI/assets/34473004/f7388b8f-0fdd-4475-81f3-a1d684e8b485">


#### Fetch some random data for sentimental score for OpenAI with Prompt
```
./instaSentimentalScore.py --master-password masterpassword -aitype openai
```
<img width="1223" alt="image" src="https://github.com/aseelert/SingleStoreAI/assets/34473004/c6e81d46-7851-477f-b38e-5b541415d816">








