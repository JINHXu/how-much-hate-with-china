#!/usr/bin/env python3
# Jinghua Xu

import json
class Keys:

    """The Keys class represents the Twitter API keys/tokens/credentials: consumer_key, consumer_secret, access_token, access_secret."""
    def __init__(self, keysfile):
        """ 
        Constructor for Keys. Read the keys/tokens/credentials from a JSON file stores them.

        Parameters
        ----------
        keysfile : String
            The file path of the JSON file stores the keys/tokens/credentials.
        """
        with open(keysfile, 'r', encoding='utf-8') as f:

            data = json.load(f)

            self._consumerKey = data['consumer key']
            self._consumerSecret = data['consumer secret']
            self._accessToken = data['access token']
            self._accessSecret = data['access secret']

    @property
    def consumer_key(self):
        return self._consumerKey
    @property
    def consumer_secret(self):
        return self._consumerSecret
    @property
    def access_token(self):
        return self._accessToken
    @property
    def access_secret(self):
        return self._accessSecret
