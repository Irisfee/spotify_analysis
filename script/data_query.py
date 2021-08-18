# -*- coding: utf-8 -*-
"""
Query information of songs of a certain genre from Spotify API.

author: Yufei Zhao
date: 2021.7.27
"""
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import yaml
import pandas as pd
from pathlib import Path
import pickle
from util import save_csv, save_pickle

class Spotify_query(object):
    def __init__(self, genre):
        # Basic directories
        self.base_dir = Path('.')
        self.data_dir = self.base_dir.joinpath("data")
        self.credential()
        self.genre = genre
        
    def credential(self):
        '''Authorization Code Flow'''
        cred = self.base_dir.joinpath('credential.yaml')
        with open(cred, "r") as f: 
            cred = yaml.load(f, Loader=yaml.CLoader)
        client_credentials_manager = SpotifyClientCredentials(
            client_id = cred['id'],
            client_secret = cred['secret'])
        self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        
    def get_artists_genre(self):
        """
        Query all artists info from a certain genre.
        Save the artist's name, id, followers into a dataframe.
        """
        artist = []
        
        # query the total number of artists
        at = self.sp.search(q='genre:'+self.genre, limit=50, type='artist')['artists']
        print('Total number of artists for '+self.genre+': '+str(at['total']))
        
        # query all artists info
        if at['total']>50:
            for i in range(0,at['total'],50):
                artist+=self.sp.search(q='genre:'+self.genre,limit=50, type='artist', offset=i)['artists']['items']
                
        # create the df of artist id, name, popularity
        artist_df = []
        for i in artist:
            d = {
                'id': i['id'],
                'name': i['name'],
                'followers': i['followers']['total'],
                'at_popularity': i['popularity']
            }
            artist_df.append(d)
        artist_df = pd.DataFrame(artist_df)
        
        # save the artist info
        fid = self.data_dir.joinpath(f'artists_genre-{self.genre}.tsv')
        save_csv(artist_df, fid)
        
        return artist_df
    
    def get_albums_artists(self, artist_df):
        """
        Query all album info from a certain artist.
        Save the artist's name, id, album_id into a dataframe.
        """
        album_df = []
        for index, row in artist_df.iterrows():
            # get all album info of an artist
            album_ar = []
            al = self.sp.artist_albums(row['id'], limit=50)
            print('Total number of albums for '+row['name']+': '+str(al['total']))
            if al['total'] > 50:
                for i in range(0, al['total'], 50):
                    album_ar += self.sp.artist_albums(row['id'], limit=50, offset=i)['items']
            else:
                album_ar += self.sp.artist_albums(row['id'], limit=50)['items']
            # create the df of artist id, album id    
            for i in album_ar:
                d = {
                    'id': row['id'],
                    'name': row['name'],
                    'album_id': i['id'],
                    'release_date': i['release_date']
                }
                album_df.append(d)    
        album_df = pd.DataFrame(album_df)
        # save data
        fid = self.data_dir.joinpath(f'artists_album_genre-{self.genre}.tsv')
        save_csv(album_df, fid)
        
        return album_df
    
    def get_songs_album(self, album_df):
        """
        Query all song basic and audio info from a certain album.
        Save the result into a dataframe.
        """
        song_df = []
        # get song info of each album
        for index, row in album_df.iterrows():
            song_al_df = []

            # get the song ids of each album
            song_al = self.sp.album_tracks(row['album_id'])['items']
            for i in song_al:
                d = {
                    'id': row['id'],
                    'name': row['name'],
                    'album_id': row['id'],
                    'release_date': row['release_date'],
                    'song_id': i['id'],
                    'preview_url': i['preview_url']
                }
                song_al_df.append(d)
            song_al_df = pd.DataFrame(song_al_df)

            # get the basic info of each song of each album
            tr_al_df = []
            song_id_list = song_al_df['song_id'].to_list()
            tr_al = self.sp.tracks(song_id_list)['tracks']
            for i in tr_al:
                d = {
                    'song_id': i['id'],
                    'popularity': i['popularity']
                }
                tr_al_df.append(d)
            tr_al_df = pd.DataFrame(tr_al_df)

            # get the audio features of each song of each album
            ft_al = self.sp.audio_features(song_id_list)
            # remove potential none item in the list
            ft_al = list(filter(None.__ne__, ft_al))
            if not ft_al:
                ft_al_df = pd.DataFrame({"song_id": song_id_list})
            else:
                ft_al_df = pd.DataFrame(ft_al)[['danceability', 'energy', 'key', 'loudness', 'speechiness',
                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'id', 'duration_ms']]
                ft_al_df = ft_al_df.rename(columns={"id": "song_id"})

            # merge tables
            song_al_df = song_al_df.merge(tr_al_df, how='left', on='song_id')
            song_al_df = song_al_df.merge(ft_al_df, how='left', on='song_id')
            song_df.append(song_al_df)
            print('Get all songs info of '+row['name']+', '+row['album_id'])
        song_df = pd.concat(song_df)

        fid = self.data_dir.joinpath(f'songs_genre-{self.genre}.pkl')
        save_pickle(song_df, fid)
        
        return song_df
    
def update_data(genre='mandopop'):
    spf = Spotify_query(genre)
    # read in artist info
    
    #upate artists info 
    artist_df = spf.get_artists_genre()
    #update albums info of artists
    album_df = spf.get_albums_artists(artist_df)
    #update songs info of albums
    spf.get_songs_album(album_df)

if __name__=="__main__":
   
    update_data(genre='mandopop')
