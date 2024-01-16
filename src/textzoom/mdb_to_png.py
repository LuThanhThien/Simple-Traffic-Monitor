import pyodbc

def main():
   # Adjust the connection string based on your database configuration
   conn_str = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=C:\Users\USER\Projects\20231019-traffic-management\src\assets\data\TextZoom\test\easy\data.mdb;'
   conn = pyodbc.connect(conn_str)
   cursor = conn.cursor()

   query = 'SELECT image_data FROM images'
   cursor.execute(query)

   from PIL import Image
   import io

   for row in cursor.fetchall():
      image_data = row.image_data
      image = Image.open(io.BytesIO(image_data))

      # Process the image as needed (e.g., save to a file)
      image.save('output_image.jpg')
      break

   cursor.close()
   conn.close()

if __name__ == '__main__':
   main()