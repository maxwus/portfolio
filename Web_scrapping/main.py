from bs4 import BeautifulSoup
import requests
from datetime import datetime
import os.path
import csv
from tqdm import tqdm
import pandas as pd
import time

file_path = 'C:/Users/maxog/Desktop/Everything/Python/Web_scrapping/logs/'
date = datetime.now()
date = date.strftime('%d_%m_%y')

def sort_tags_logic(new, sale, popular, preorder):
    if tag.text == 'Novinka':                    
        new = True           
        
    elif tag.text == 'Výprodej':
        sale = True
        
    elif tag.text == 'Populární':
        popular = True
        
    elif tag.text == 'Předobjednávka':
        preorder = True
    return new, sale, popular, preorder

def sort_tags():
    
    global tag   
    new = False
    sale = False
    popular = False
    preorder = False 
    
    tag = inf.find('div', class_ = 'LST_lab')
    new, sale, popular, preorder = sort_tags_logic(new, sale, popular, preorder)
                    
    tag = tag.find_next_sibling()
    new, sale, popular, preorder = sort_tags_logic(new, sale, popular, preorder)
                            
    tag = tag.find_next_sibling()     
    new, sale, popular, preorder = sort_tags_logic(new, sale, popular, preorder)
                          
    tag = tag.find_next_sibling()
    new, sale, popular, preorder = sort_tags_logic(new, sale, popular, preorder)
              
    return new, sale, popular, preorder  
  
#%% ONE PIECE TRACKER
op_url = 'https://www.nejlevnejsi-knihy.cz/vyhledavani_1.html?OP=1&q=one%20piece%20omnibus'
OP_dict_list = []

file_name = f'one_piece_omnibus/{date}.csv'
full_path = os.path.join(file_path, file_name)

with  open(full_path, 'w', newline='', encoding = 'UTF8') as OP_file:
    keys = ['full_title','series', 'volume_number',  'price','currency', 'availability']
    
    op_output = csv.writer(OP_file)
    op_output.writerow(keys)
       
    for  i in (1,2):
        url = op_url[:45] + str(i) + op_url[45 + 1:]
    
        html_text = requests.get(url).text   
        soup = BeautifulSoup(html_text, 'lxml')
        OP_omnibuses = soup.find_all('div', class_ = 'LST_inf')
        OP_prices = soup.find_all('div', class_ = "LST_buy")
        OP_availability = soup.find_all('div', class_ = 'buy') 
       
        for omnibus,price, availabity in zip(OP_omnibuses, OP_prices, OP_availability):      
            listed_name = omnibus.find('h3').text
            title = listed_name[:-9]
            volume_number = listed_name[-2:]
            full_price = price.find('p' ).text
            price = full_price[:-3]
            currency = full_price[-2:]
            availability = availabity.find('a').text
            
            
            values = [listed_name, title, volume_number, price, currency, availability]
            
            op_output.writerow(values)

#%% all books

czech = True 
stop_loop = False

for i in range(2): 
    
    if czech:
        url_root = 'https://www.nejlevnejsi-knihy.cz/knihy-v-cestine_'
        file_name = f'databases/nejlevnejsi_knihy/czech/{date}.csv'    
        print('\n Collecting czech books')
    else:
        url_root = 'https://www.nejlevnejsi-knihy.cz/knihy-v-anglictine_'
        file_name = f'databases/nejlevnejsi_knihy/english/{date}.csv'
        print('\n Collecting english books')
        
    full_path = os.path.join(file_path,file_name)
    
    with open(full_path, 'w', newline = '', encoding = 'windows-1250') as database:
        #header of the database
        keys = ['title', 'info', 'available',  'price', 'currency', 'binding', 'language', 'new?', 'sale?', 'popular?', 'preorder']
        
        database_output = csv.writer(database)
        database_output.writerow(keys)
           
        #initiation of beutiful soup module
        if czech:
            url_pages = 'https://www.nejlevnejsi-knihy.cz/knihy-v-cestine_1.html'
            html_text = requests.get(url_pages).text
            soup = BeautifulSoup(html_text, 'lxml')
            number_of_pages = soup.find('p', class_ = 'LST_pag').text.replace(' ', '')[-4:]                
        else:
            url_pages = 'https://www.nejlevnejsi-knihy.cz/knihy-v-anglictine_1.html'
            html_text = requests.get(url_pages).text
            soup = BeautifulSoup(html_text, 'lxml')
            number_of_pages = soup.find('p', class_ = 'LST_pag').text.replace(' ', '')[-6:]
            
        number_of_pages = int(number_of_pages) + 1     
        start_time = datetime.now()
        removed = [] #used for entries that cannot be coded in 'windows-1250
        
    
        for i in tqdm(range(1,number_of_pages)):
            url =  f'{url_root}{i}.html'
        
            # connecting to the html page
            html_text = ''
            while html_text == '':
                try:
                    html_text = requests.get(url)              
                    break
                except:
                    print("Connection refused by the server..")
                    time.sleep(5)
                    print('Trying to reconnect')
                     
            soup = BeautifulSoup(html_text.text, 'lxml')
                          
            all_inf_tab = soup.find_all('div', class_ = 'LST_inf' )
            all_buy_tab = soup.find_all('div', class_ = 'LST_buy')
            
            
            #checks if scrapped page was empty
            #necessery, because number of pages in english is in order of e6
            # after page 500 there are empty pages on the server
            # possible mistake on their part, might be fixed in the future       
            if all_inf_tab == []:
                stop_loop = True
                print(f'\n Disconecting from the server, empty page {i} reached')
                break
            
        
            # TODO Create individual filtering for year, publisher author, problem with differnet amount of words
            #as a consequence of static html url code it might be possible     
            # to get link to individual page in source code, possible TODO
            
        
            for inf, buy in zip(all_inf_tab, all_buy_tab):
                
                title_name = inf.find('h3').text.replace('│', '') 
                
                info = inf.find('h4').text.replace('|','').replace('′','\'')
                available = inf.find('div', class_ = 'LST_addToCart')
                
                if pd.isna(available) == True:
                    available = 'ocakavany dotisk'
                else:
                    available = 'skladem'
                
                full_price = buy.find('p' ).text
                price = full_price[:-3]
                currency = full_price[-2:]
                
                
                #sorting optional tags
                try:
                    new, sale, popular, preorder = sort_tags()
                except:
                    pass
                    
                
                language = inf.find('p').text
                try:
                    binding = inf.find('p').next_sibling.next_sibling.text[7:]
                except:
                    biding = False
                           
                values = [title_name ,info, available, price, currency, binding, language,new, sale, popular, preorder]   
                
                #saving uncompatible rows into variable remove
                #TODO save values to csv file with different encryption
                # print(title_name)
               
                try:
                    database_output.writerow(values)                  
                except:
                    removed.append(values)
                                    
        # in case empty page was scrapped, program switches to english books 
        if stop_loop == True and czech == True:
            czech = False
            stop_loop = False
            continue
        
        #escpaes the scrapping process if empty english page was encountered
        elif stop_loop == True and czech == False:
            break
            
    
end_time = datetime.now()
dt = end_time - start_time
seconds_in_day = 24 * 60 * 60
divmod(dt.days * seconds_in_day + dt.seconds, 60) #output (mins, seconds)
