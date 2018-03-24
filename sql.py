"""
automated SQL query to SDSS database for star data
"""

# !pip install mechanize
import mechanicalsoup
#from StringIO import StringIO

# dr = [10,12,13,14] # only works for these dr{vals}
dr = 14
url = "http://skyserver.sdss.org/dr{}/en/tools/search/sql.aspx".format(dr)
br = mechanicalsoup.StatefulBrowser()

def SDSS_select(sql):
    '''
    input:  string with a valid SQL query
    output: csv
    source: http://balbuceosastropy.blogspot.com/2013/10/an-easy-way-to-make-sql-queries-from.html
    '''
    br.open(url)
    br.select_form("[name=sql]")
    br['cmd'] = sql
    br["format"]="csv"
    response = br.submit_selected()
    return response.text
    
def writer(name, data):
    # writes data to a file
    f = open(name, 'w')
    f.write(data)
    f.close()
    return writer

# n = number of objects
n = 50000
# s = SQL query
s = "SELECT TOP {} \
        p.u,p.g,p.r,p.i,p.z,s.subClass \
    FROM PhotoObj p \
    JOIN SpecObj AS s ON p.objID = s.bestObjID \
    WHERE p.type = 6 \
    AND p.clean = 1 \
    AND s.Class = 'STAR'".format(n)

SDSS = SDSS_select(s)

#print SDSS
filename = 'data.csv'
writer(filename, SDSS)