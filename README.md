Finished !
### Output JSON Format
```
{
  "status": true,
  "code": "S100",
  "message": "موفق",
  "data": {
    "request_id": "6",
    "video_path": null,
    "priority": 1,
    "result": {
      "total_frames": 150,
      "frames_processed": 15,
      "emotion_percentages": {
        "affection": 0,
        "anger": 0,
        "annoyance": 0,
        "anticipation": 33.33333333333333,
        "aversion": 0,
        "confidence": 100,
        "disapproval": 0,
        "disconnection": 0,
        "disquietment": 0,
        "doubt/Confusion": 0,
        "embarrassment": 0,
        "engagement": 13.333333333333334,
        "esteem": 0,
        "excitement": 80,
        "fatigue": 0,
        "fear": 0,
        "happiness": 100,
        "pain": 0,
        "peace": 0,
        "pleasure": 80,
        "sadness": 0,
        "sensitivity": 0,
        "suffering": 0,
        "surprise": 0,
        "sympathy": 0,
        "yearning": 0
      }
    },
    "webhook_retry_count": 0,
    "itime": "2025-07-28T11:35:15.247088",
    "descr": null,
    "id": "b399e837-1f35-4aae-be6d-8b8ab48b1c6f",
    "status": 2,
    "error": null,
    "webhook_status_code": null,
    "utime": "2025-07-28T11:35:20.215464"
  }
}
```
<header>
  
  # VideoToTxt
  
  </header>
  
**The Diagram**:

<div align="center">
  <img src="https://github.com/user-attachments/assets/241e9a98-3e15-4745-91dd-6e5c8b34ebe0" alt="VideoToTxt(Mohamad Taghizadeh)" width="500" height="300">
</div>

**Here is the detailed structure**:
![VideoToTxt diagram(Mohamad Taghizadeh)](https://github.com/user-attachments/assets/df9aa459-80b4-4bcb-88a2-c8a98b98f364)

**Body Language and Emotion Recognition - Video to Text (JSON)**

>**26 categories**

1. **Affection**: fond feelings; love; tenderness
2. **Anger**: intense displeasure or rage; furious; resentful
3. **Annoyance**: bothered by something or someone; irritated; impatient; frustrated
4. **Anticipation**: state of looking forward; hoping on or getting prepared for possible future events
5. **Aversion**: feeling disgust, dislike, repulsion; feeling hate
6. **Confidence**: feeling of being certain; conviction that an outcome will be favorable; encouraged; proud
7. **Disapproval**: feeling that something is wrong or reprehensible; contempt; hostile
8. **Disconnection**: feeling not interested in the main event of the surrounding; indifferent; bored; distracted
9. **Disquietment**: nervous; worried; upset; anxious; tense; pressured; alarmed
10. **Doubt/Confusion**: difficulty to understand or decide; thinking about different options
11. **Embarrassment**: feeling ashamed or guilty
12. **Engagement**: paying attention to something; absorbed into something; curious; interested
13. **Esteem**: feelings of favourable opinion or judgement; respect; admiration; gratefulness
14. **Excitement**: feeling enthusiasm; stimulated; energetic
15. **Fatigue**: weariness; tiredness; sleepy
16. **Fear**: feeling suspicious or afraid of danger, threat, evil or pain; horror
17. **Happiness**: feeling delighted; feeling enjoyment or amusement
18. **Pain**: physical suffering
19. **Peace**: well being and relaxed; no worry; having positive thoughts or sensations; satisfied
20. **Pleasure**: feeling of delight in the senses
21. **Sadness**: feeling unhappy, sorrow, disappointed, or discouraged
22. **Sensitivity**: feeling of being physically or emotionally wounded; feeling delicate or vulnerable
23. **Suffering**: psychological or emotional pain; distressed; anguished
24. **Surprise**: sudden discovery of something unexpected
25. **Sympathy**: state of sharing others emotions, goals or troubles; supportive; compassionate
26. **Yearning**: strong desire to have something; jealous; envious; lust



<header>
  
  # Folder structure:
  
  </header>



>Folder structure:
```
Computer_vision
          ├── Models
          |   ├── VideoToTxt
          |   ├── 
          |   ├──
          |   ..
          |   ├── .gitignore
          ├── Outputs
          |   ├── VideoToTxt
          |       ├── .gitignore
          |   ├── 
          |   ├──
          |   ..
          ├── Samples
          ├── Scource code
              ├── VideoToTxt
                  ├── VideoToTxt ─── src
                  ├── .gitignore      |
                                      ├── engine
                                      |    ├── confing
                                      |           ├── conf-dev.yml
                                      |           ├── conf-prod.yml
                                      |           ├── conf_handler.yml
                                      |    ├── core
                                      |           ├── queue_utils.py
                                      |    ├── Dockerfile
                                      |    ├── cmd.sh
                                      |    ├── main.py
                                      |    ├── requirements.txt
                                      |    ├── utils.py
                                      |    ├── version.py
                                      ├── backend
                                      |    ├── confing
                                      |           ├── conf-dev.yml
                                      |           ├── conf-prod.yml
                                      |           ├── conf_handler.yml
                                      |    ├── core
                                      |           ├── base.py
                                      |           ├── messages.py
                                      |           ├── queue_utils.py
                                      |           ├── utils.py
                                      |           ├── webhook_handler.py
                                      |    ├── dbutils
                                      |           ├── crud.py
                                      |           ├── database.py
                                      |           ├── models.py
                                      |           ├── schemas.py
                                      |    ├── Dockerfile
                                      |    ├── cmd.sh
                                      |    ├── mainapi.py
                                      |    ├── requirements.txt
                                      |    ├── version.py
                                      ├── rabbitmq
                                      |    ├── confing
                                      |           ├── rabbitmq.conf
                                      ├── template
                                      |           ├── env
                                      ├── .gitignore
                                      ├── docker-compose.yml
```

<header>
  
  # Run
  
<header>

> Run

```
!python main.py --mode vedio --inference_file ./assets/raw.mp4 --experiment_path ./proj/debug_ex
```

<header>
  
  # Result:
  

https://github.com/user-attachments/assets/bc226bb1-9fec-4600-8522-a9920613c885


  
  </header>


# Plot categories and vad

>categorie
![category_detection](https://github.com/user-attachments/assets/95c5f70c-d4ac-48f7-82c1-3f831bb62173)


>vad
![vad_values](https://github.com/user-attachments/assets/aa0f653c-0c90-475e-9569-edae289df024)



## Start to local test
### Terminal 1: RabbitMQ
```docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4.0.6-management```

### Terminal 2: Engine 
```cd ".\src\engine"```
```python main.py```

### Terminal 3: Backend
```cd ".\src\backend"```
```uvicorn mainapi:app --host localhost --port 8000```

--------------------------------------------------------------------------------------
## Start to Dockerize

Updating...
> Build Image
```
docker-compose up --build
```

> Test Locally. Modify conf_dev.yml (Backend & Engine)
```
QUEUE_CONNECTION: amqp://guest:guest@localhost/  
```

Then, install Rabbitmq
```
docker run -p 5672:5672 rabbitmq
```
Finally, run Backend and Engine
```
uvicorn mainapi:app --host 0.0.0.0 --port 8000
```

