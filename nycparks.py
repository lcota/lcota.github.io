Creating a Python script to log in to a website and automatically perform actions like reserving a court involves using web automation tools such as Selenium. Here’s a step-by-step guide and a script that logs into the NYC Parks tennis reservation system and attempts to reserve court 6. Note that you will need to fill in your login credentials and possibly modify the script to suit any specific details of the website.

First, ensure you have Selenium and the appropriate WebDriver installed. You can install Selenium using pip:

```bash
pip install selenium
```

Here's a basic outline of the script:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Replace these with your actual login credentials
username = "your_username"
password = "your_password"

# Set up the WebDriver (Make sure the path to your WebDriver is correct)
driver_path = 'path_to_your_webdriver'
driver = webdriver.Chrome(executable_path=driver_path)

try:
    # Navigate to the login page
    driver.get("https://www.nycgovparks.org/tennisreservation/availability/11")

    # Wait until the login form is available and fill it in
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "login")))
    
    # Enter username
    driver.find_element(By.ID, "login").send_keys(username)
    
    # Enter password
    driver.find_element(By.ID, "password").send_keys(password)
    
    # Submit the form
    driver.find_element(By.ID, "password").send_keys(Keys.RETURN)

    # Wait until the reservation page is loaded and the courts are displayed
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@class='court-container']")))

    # Find and click on court 6
    court_6 = driver.find_element(By.XPATH, "//div[@class='court-container' and contains(text(), 'Court 6')]")
    court_6.click()
    
    # Confirm reservation (modify this according to the actual flow of the site)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "confirm-reservation")))
    driver.find_element(By.ID, "confirm-reservation").click()

    print("Court 6 reserved successfully!")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the WebDriver
    driver.quit()
```

### Notes:
1. **WebDriver Path**: Replace `'path_to_your_webdriver'` with the actual path to your WebDriver executable (e.g., chromedriver).
2. **Selectors**: The `By.ID` and `By.XPATH` selectors in the script need to be accurate according to the website's HTML structure. You might need to inspect the website elements and adjust these accordingly.
3. **Wait Times**: Adjust the wait times (currently set to 10 seconds) based on your network speed and the responsiveness of the website.
4. **Error Handling**: The script includes basic error handling. For a production script, you might want to enhance this further.

Ensure you test the script thoroughly and make adjustments based on the actual flow and elements of the website.