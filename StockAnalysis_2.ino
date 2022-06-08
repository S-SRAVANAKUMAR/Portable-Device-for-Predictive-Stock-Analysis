#include "ThingSpeak.h" 
#include <ESP8266WiFi.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
const char ssid[] = "Chitravelan";  
const char pass[] = "Qwerty2426.";   

int statusCode = 0;

WiFiClient  client;
LiquidCrystal_I2C lcd(0x27, 16, 2);

//-------Channel Details - Apple---------//
unsigned long Cid1 = 1700245;            
const char * ReadAPIKey1 = "ZA4FI7MH2VJ9D16R"; 

//-------Channel Details - Google---------//
unsigned long Cid2 = 1700241;            
const char * ReadAPIKey2 = "6BKSY4REPIWFBOZ1"; 

//-------Channel Details - Microsoft---------//
unsigned long Cid3 = 1709210;            
const char * ReadAPIKey3 = "4C61WM21LMGDQZK2"; 

//-------Channel Details - Amazon---------//
unsigned long Cid4 = 1700237;            
const char * ReadAPIKey4 = "JUX9NP24N8KI6PT7"; 

const int FieldNumber1 = 1; //Forecasted open price
const int FieldNumber2 = 2; //Risk Analysis
const int FieldNumber3 = 3; //Trend
const int FieldNumber4 = 4; //Company
const int FieldNumber5 = 5; //News

void setup()
{
  WiFi.mode(WIFI_STA);
  ThingSpeak.begin(client);
  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Display Test:OK");
  delay(1000);
}

void loop()
{
  //----------------- Network -----------------//
  if (WiFi.status() != WL_CONNECTED)
  {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Connecting to ");
    lcd.setCursor(0, 1);
    lcd.print(ssid);
    delay(1000);
    while (WiFi.status() != WL_CONNECTED)
    {
      WiFi.begin(ssid, pass);
      delay(5000);
    }
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Conn.. to Wi-Fi");
    lcd.setCursor(0, 1);
    lcd.print("Succesfully.");
    delay(1000);
    lcd.clear();
  }
  //End of Network connection

  //---------------- Channel 1 - APPLE ----------------//
  lcd.setCursor(5, 0);
  lcd.print("APPLE");
  ForecastedOpenPrice(Cid1, FieldNumber1, ReadAPIKey1);
  RiskAnalysis(Cid1, FieldNumber2, ReadAPIKey1);
  Trend(Cid1, FieldNumber3, ReadAPIKey1);
  //Company(Cid1, FieldNumber4, ReadAPIKey1);
  News(Cid1, FieldNumber5, ReadAPIKey1);
  delay(2000);
  lcd.clear();


  //---------------- Channel 2 - GOOGLE ----------------//
  lcd.setCursor(5, 0);
  lcd.print("GOOGLE");
  ForecastedOpenPrice(Cid2, FieldNumber1, ReadAPIKey2);
  RiskAnalysis(Cid2, FieldNumber2, ReadAPIKey2);
  Trend(Cid2, FieldNumber3, ReadAPIKey2);
  //Company(Cid2, FieldNumber4, ReadAPIKey2);
  News(Cid2, FieldNumber5, ReadAPIKey2);
  delay(2000);
  lcd.clear();


 //---------------- Channel 3 - MICROSOFT ----------------//
  lcd.setCursor(4, 0);
  lcd.print("MICROSOFT");
  ForecastedOpenPrice(Cid3, FieldNumber1, ReadAPIKey3);
  RiskAnalysis(Cid3, FieldNumber2, ReadAPIKey3);
  Trend(Cid3, FieldNumber3, ReadAPIKey3);
  //Company(Cid3, FieldNumber4, ReadAPIKey3);
  News(Cid3, FieldNumber5, ReadAPIKey3);
  delay(2000);
  lcd.clear();


  //---------------- Channel 4 - AMAZON ----------------//
  lcd.setCursor(5, 0);
  lcd.print("AMAZON");
  ForecastedOpenPrice(Cid4, FieldNumber1, ReadAPIKey4);
  RiskAnalysis(Cid4, FieldNumber2, ReadAPIKey4);
  Trend(Cid4, FieldNumber3, ReadAPIKey4);
  //Company(Cid4, FieldNumber4, ReadAPIKey4);
  News(Cid4, FieldNumber5, ReadAPIKey4);
  delay(2000);
  lcd.clear();

  /*lcd.setCursor(0, 0);
  lcd.print("Company Insights");
  lcd.setCursor(0, 1);
  lcd.print("Choose 0-Apple, 1-Google, 2-Microsoft, 3-Amazon");
  for(int i=0 ; i<35 ; i++)
  {
    lcd.scrollDisplayLeft();
    delay(500);
  }*/
}

void ForecastedOpenPrice(long cid, const int fieldno, const char * apikey)    //FORECASTED OPEN PRICE 
{
  long fop = ThingSpeak.readLongField(cid, fieldno, apikey);
  statusCode = ThingSpeak.getLastReadStatus();
  if (statusCode == 200)
  {
    lcd.setCursor(0, 1);
    lcd.print("F.Open Price:");
    lcd.print(fop);
  }
  else
  {
    lcd.clear();
    lcd.setCursor(0, 1);
    lcd.print("Unable to read");
  }
  lcd.print("                ");
  delay(1000);
}

void RiskAnalysis(long cid, const int fieldno, const char * apikey)     //RISK ANALYSIS
{
  long risk = ThingSpeak.readLongField(cid, fieldno, apikey);  
  statusCode = ThingSpeak.getLastReadStatus();
  if (statusCode == 200)
  {
    lcd.setCursor(0, 1);
    lcd.print("Risk:");
    lcd.print(risk);
  }
  else
  {
    lcd.clear();
    lcd.setCursor(0, 1);
    lcd.print("Unable to read");
  }
  lcd.print("                ");
  delay(1000);
}

void Trend(long cid, const int fieldno, const char * apikey)     //TREND
{
  long tr = ThingSpeak.readLongField(cid, fieldno, apikey);  
  statusCode = ThingSpeak.getLastReadStatus();
  if (statusCode == 200)
  {
    lcd.setCursor(0, 1);
    lcd.print("Upcoming Trends");
    delay(1000);
    lcd.print("                ");
    delay(100);
    if(tr>=0)
    {
      lcd.setCursor(0, 1);
      lcd.print("Overall +ve");
    }
    else
    {
      lcd.setCursor(0, 1);
      lcd.print("Overall -ve");
    }
  }
  else
  {
    lcd.clear();
    lcd.setCursor(0, 1);
    lcd.print("Unable to read");
  }
  lcd.print("                ");
  delay(1000);
}

void Company(long cid, const int fieldno, const char * apikey)     //COMPANY
{
  long com = ThingSpeak.readLongField(cid, fieldno, apikey);  
  statusCode = ThingSpeak.getLastReadStatus();
  if (statusCode == 200)
  {
    lcd.setCursor(0, 1);
    lcd.print("Company:");
    lcd.print(com);
  }
  else
  {
    lcd.clear();
    lcd.setCursor(0, 1);
    lcd.print("Unable to read");
  }
  lcd.print("                ");
  delay(1000);
}

void News(long cid, const int fieldno, const char * apikey)     //NEWS
{
  long n = ThingSpeak.readLongField(cid, fieldno, apikey);  
  statusCode = ThingSpeak.getLastReadStatus();
  if (statusCode == 200)
  {
    lcd.setCursor(0, 1);
    lcd.print("News- Suggestion");
    delay(1000);
    lcd.print("                ");
    delay(100);
    if(n==0)
    {
      lcd.setCursor(0, 1);
      lcd.print("Buy some shares");
    }
    else
    {
      lcd.setCursor(0, 1);
      lcd.print("Sell some shares");
    }
  }
  else
  {
    lcd.clear();
    lcd.setCursor(0, 1);
    lcd.print("Unable to read");
  }
  lcd.print("                ");
  delay(1000);
}
