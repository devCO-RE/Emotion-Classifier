# -*- coding: utf-8 -*- 
import pickle
import csv
import os

import google.oauth2.credentials

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

CLIENT_SECRETS_FILE=os.environ["YOUTUBE_V3_CLIENT_SECRET_FILE"]
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


#api를 사용할 수 있는 서비스를 빌드한다.
def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()
 
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
 
    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials)

#영어 댓글 너무 많으면 거른다... 영어인지 간단 체크
def is_english_comment(c):
    if(ord(c) >= 65 and ord(c) <= 90):                  
        return True
    elif(ord(c) >= 97 and ord(c) <= 122):               
        return True
    else:
        return False

def write_to_csv(comments):
    with open('datas/youtube_comments18.csv', 'w') as comments_file:
        comments_file.writelines('Comment\n')
        #comments_writer = csv.writer(comments_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) # 하나하나 토큰화 나중에 할거면 애초에 이렇게 저장하는거도 좋은 방법인듯(한음절씩 저장됨)
        # comments_writer.writerow('Comment')
        for row in comments:
            # convert the tuple to a list and write to the output file
            comments_file.writelines("\""+row+"\"")
            comments_file.writelines('\n')
            #comments_writer.writerow(row)


def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()
    
    comments_cnt=0
    while results and comments_cnt<5000:
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            if not is_english_comment(comment[0]) and len(comment)<=500:
                comments.append(comment)
                comments_cnt=comments_cnt+1
 
        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.commentThreads().list(**kwargs).execute()
            
        else:
            break
 
    write_to_csv(comments)



if __name__ == '__main__':
    # When running locally, disable OAuthlib's HTTPs verification. When
    # running in production *do not* leave this option enabled.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    service = get_authenticated_service()
    video_id = input('Enter a video ID: ')
    try:
        get_video_comments(service, part='snippet', videoId=video_id, order='relevance', textFormat='plainText')
    
    except HttpError as e:
        print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
   
 

