"""
Synthetic Spider + BIRD benchmark — mode="llm" with gpt-4o-mini.

Creates 50-example synthetic datasets in Spider/BIRD file format, runs
them through text2ql's native benchmark runner, and prints a full report.

Usage:
    OPENAI_API_KEY=sk-... python run_llm_benchmark.py
"""
from __future__ import annotations

import json
import importlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

# Reused fixture literals extracted for Sonar S1192 (duplicate string literal).
ALAMEDA_CITY_UNIFIED = "Alameda City Unified"
ALAMEDA_HIGH = "Alameda High"
ENCINAL_HIGH = "Encinal High"
ISLAND_HIGH = "Island High"
ALPINE_COUNTY_UNIFIED = "Alpine County Unified"
ALPINE_CREST = "Alpine Crest"
SEASON_2008_2009 = "2008/2009"

# ---------------------------------------------------------------------------
# 50 Spider-style examples across 5 databases
# ---------------------------------------------------------------------------

SPIDER_DBS = {
    "concert_singer": {
        "tables": ["singer", "concert", "singer_in_concert"],
        "columns": {
            "singer": ["Singer_ID", "Name", "Country", "Song_Name", "Song_release_year", "Age", "Is_male"],
            "concert": ["concert_ID", "concert_Name", "Theme", "Stadium_ID", "Year"],
            "singer_in_concert": ["concert_ID", "Singer_ID"],
        },
        "foreign_keys": [
            ("singer_in_concert", "Singer_ID", "singer", "Singer_ID"),
            ("singer_in_concert", "concert_ID", "concert", "concert_ID"),
        ],
        "data": {
            "singer": [
                (1, "John", "USA", "Blue Sky", "2000", 35, 1),
                (2, "Mary", "UK", "Red Rose", "2005", 28, 0),
                (3, "Bob", "Canada", "Storm", "2010", 42, 1),
                (4, "Lisa", "Australia", "Rain", "2015", 25, 0),
                (5, "Tom", "USA", "Wind", "2018", 31, 1),
            ],
            "concert": [
                (1, "Spring Fest", "Pop", 1, "2019"),
                (2, "Summer Blast", "Rock", 2, "2020"),
                (3, "Winter Show", "Jazz", 1, "2021"),
            ],
            "singer_in_concert": [
                (1, 1), (1, 2), (2, 3), (3, 4), (3, 5),
            ],
        },
    },
    "car_1": {
        "tables": ["continents", "countries", "car_makers", "model_list", "car_names", "cars_data"],
        "columns": {
            "continents": ["ContId", "Continent"],
            "countries": ["CountryId", "CountryName", "Continent"],
            "car_makers": ["Id", "Maker", "FullName", "Country"],
            "model_list": ["ModelId", "Maker", "Model"],
            "car_names": ["MakeId", "Model", "Make"],
            "cars_data": ["Id", "MPG", "Cylinders", "Edispl", "Horsepower", "Weight", "Accelerate", "Year"],
        },
        "foreign_keys": [
            ("countries", "Continent", "continents", "ContId"),
            ("car_makers", "Country", "countries", "CountryId"),
            ("model_list", "Maker", "car_makers", "Id"),
        ],
        "data": {
            "continents": [(1, "america"), (2, "europe"), (3, "asia")],
            "countries": [(1, "USA", 1), (2, "Germany", 2), (3, "Japan", 3)],
            "car_makers": [(1, "ford", "Ford Motor", "1"), (2, "bmw", "BMW", "2"), (3, "toyota", "Toyota", "3")],
            "model_list": [(1, 1, "mustang"), (2, 1, "focus"), (3, 2, "3series"), (4, 3, "corolla")],
            "car_names": [(1, "mustang", "Ford Mustang"), (2, "focus", "Ford Focus"), (3, "3series", "BMW 3"), (4, "corolla", "Toyota Corolla")],
            "cars_data": [(1, 18.0, 8, 307.0, 130.0, 3504, 12.0, 1970), (2, 25.0, 4, 98.0, 90.0, 2265, 15.5, 1978), (3, 32.0, 4, 97.0, 52.0, 2130, 24.6, 1982)],
        },
    },
    "employee_hire": {
        "tables": ["employee", "shop", "hiring", "evaluation"],
        "columns": {
            "employee": ["Employee_ID", "Name", "Age", "City"],
            "shop": ["Shop_ID", "Name", "Location", "Open_Year", "Number_products"],
            "hiring": ["Shop_ID", "Employee_ID", "Start_from", "Is_full_time"],
            "evaluation": ["Employee_ID", "Year_awarded", "Bonus"],
        },
        "foreign_keys": [
            ("hiring", "Shop_ID", "shop", "Shop_ID"),
            ("hiring", "Employee_ID", "employee", "Employee_ID"),
            ("evaluation", "Employee_ID", "employee", "Employee_ID"),
        ],
        "data": {
            "employee": [(1, "Alice", 30, "New York"), (2, "Bob", 45, "LA"), (3, "Charlie", 28, "Chicago"), (4, "Diana", 35, "Houston"), (5, "Eve", 52, "Phoenix")],
            "shop": [(1, "TechStore", "Mall A", 2010, 500), (2, "GadgetHub", "Mall B", 2015, 300), (3, "ElecWorld", "Mall C", 2008, 800)],
            "hiring": [(1, 1, "2015-01-01", 1), (1, 2, "2016-03-15", 0), (2, 3, "2017-06-01", 1), (3, 4, "2019-01-10", 1), (3, 5, "2020-07-22", 0)],
            "evaluation": [(1, 2019, 1000.0), (1, 2020, 1500.0), (2, 2020, 800.0), (3, 2021, 1200.0)],
        },
    },
    "student_transcript": {
        "tables": ["addresses", "students", "courses", "student_course_attendance", "student_course_registrations"],
        "columns": {
            "addresses": ["address_id", "line_1", "line_2", "city", "state", "country"],
            "students": ["student_id", "first_name", "last_name", "address_id", "date_of_birth", "current_address_id"],
            "courses": ["course_id", "course_name", "course_description", "other_details"],
            "student_course_attendance": ["student_id", "course_id", "date_of_attendance"],
            "student_course_registrations": ["student_id", "course_id", "registration_date"],
        },
        "foreign_keys": [
            ("students", "address_id", "addresses", "address_id"),
            ("student_course_attendance", "student_id", "students", "student_id"),
            ("student_course_attendance", "course_id", "courses", "course_id"),
            ("student_course_registrations", "student_id", "students", "student_id"),
            ("student_course_registrations", "course_id", "courses", "course_id"),
        ],
        "data": {
            "addresses": [(1, "123 Main", None, "Boston", "MA", "USA"), (2, "456 Oak", None, "Austin", "TX", "USA")],
            "students": [(1, "John", "Smith", 1, "2000-05-01", 1), (2, "Jane", "Doe", 2, "2001-08-12", 2), (3, "Mike", "Jones", 1, "1999-12-25", 1)],
            "courses": [(1, "Math 101", "Basic Algebra", None), (2, "CS 201", "Intro to Programming", None), (3, "Physics 101", "Mechanics", None)],
            "student_course_attendance": [(1, 1, "2022-01-10"), (1, 2, "2022-01-10"), (2, 2, "2022-01-10"), (3, 3, "2022-01-10")],
            "student_course_registrations": [(1, 1, "2021-12-01"), (1, 2, "2021-12-01"), (2, 2, "2021-12-01"), (2, 3, "2021-12-01"), (3, 1, "2021-12-01"), (3, 3, "2021-12-01")],
        },
    },
    "poker_player": {
        "tables": ["people", "poker_player"],
        "columns": {
            "people": ["People_ID", "Nationality", "Name", "Birth_Date", "Height"],
            "poker_player": ["Poker_Player_ID", "People_ID", "Final_Table_Made", "Best_Finish", "Money_Rank", "Earnings"],
        },
        "foreign_keys": [
            ("poker_player", "People_ID", "people", "People_ID"),
        ],
        "data": {
            "people": [(1, "USA", "Alice Fox", "1980-03-12", 1.75), (2, "UK", "Bob Lane", "1975-07-22", 1.82), (3, "Canada", "Carl Ross", "1990-11-05", 1.70), (4, "Australia", "Dana Hill", "1985-01-30", 1.68), (5, "USA", "Eric Bane", "1978-09-15", 1.79)],
            "poker_player": [(1, 1, 5, 2, 10, 250000.0), (2, 2, 8, 1, 5, 500000.0), (3, 3, 3, 5, 20, 80000.0), (4, 4, 10, 1, 3, 750000.0), (5, 5, 2, 8, 15, 120000.0)],
        },
    },
}

SPIDER_EXAMPLES = [
    # concert_singer
    {"db_id": "concert_singer", "question": "How many singers are there?", "query": "SELECT count(*) FROM singer", "difficulty": "easy"},
    {"db_id": "concert_singer", "question": "What are all the names of the singers?", "query": "SELECT Name FROM singer", "difficulty": "easy"},
    {"db_id": "concert_singer", "question": "What are the names of the singers from the USA?", "query": "SELECT Name FROM singer WHERE Country = 'USA'", "difficulty": "easy"},
    {"db_id": "concert_singer", "question": "Show the name and the release year of the song by the youngest singer.", "query": "SELECT Song_Name, Song_release_year FROM singer ORDER BY Age ASC LIMIT 1", "difficulty": "medium"},
    {"db_id": "concert_singer", "question": "What is the average age of all singers from each country?", "query": "SELECT Country, AVG(Age) FROM singer GROUP BY Country", "difficulty": "medium"},
    {"db_id": "concert_singer", "question": "How many concerts are there in each year?", "query": "SELECT Year, count(*) FROM concert GROUP BY Year", "difficulty": "easy"},
    {"db_id": "concert_singer", "question": "Show names of all singers who participated in more than one concert.", "query": "SELECT T2.Name FROM singer_in_concert AS T1 JOIN singer AS T2 ON T1.Singer_ID = T2.Singer_ID GROUP BY T1.Singer_ID HAVING count(*) > 1", "difficulty": "hard"},
    {"db_id": "concert_singer", "question": "What are the names and themes of all concerts?", "query": "SELECT concert_Name, Theme FROM concert", "difficulty": "easy"},
    {"db_id": "concert_singer", "question": "How many male singers are there?", "query": "SELECT count(*) FROM singer WHERE Is_male = 1", "difficulty": "easy"},
    {"db_id": "concert_singer", "question": "What is the maximum and minimum age of all singers?", "query": "SELECT MAX(Age), MIN(Age) FROM singer", "difficulty": "easy"},
    # car_1
    {"db_id": "car_1", "question": "How many car models are there?", "query": "SELECT count(*) FROM model_list", "difficulty": "easy"},
    {"db_id": "car_1", "question": "What are the names of all car makers?", "query": "SELECT FullName FROM car_makers", "difficulty": "easy"},
    {"db_id": "car_1", "question": "What is the average mpg for all cars?", "query": "SELECT AVG(MPG) FROM cars_data", "difficulty": "easy"},
    {"db_id": "car_1", "question": "How many cars have more than 6 cylinders?", "query": "SELECT count(*) FROM cars_data WHERE Cylinders > 6", "difficulty": "easy"},
    {"db_id": "car_1", "question": "What is the maximum horsepower of all cars?", "query": "SELECT MAX(Horsepower) FROM cars_data", "difficulty": "easy"},
    {"db_id": "car_1", "question": "What are the models produced by Ford?", "query": "SELECT T2.Model FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Id = T2.Maker WHERE T1.FullName = 'Ford Motor'", "difficulty": "medium"},
    {"db_id": "car_1", "question": "How many car makers are from each country?", "query": "SELECT T2.CountryName, count(*) FROM car_makers AS T1 JOIN countries AS T2 ON T1.Country = T2.CountryId GROUP BY T1.Country", "difficulty": "medium"},
    {"db_id": "car_1", "question": "What is the average weight of cars produced in 1978?", "query": "SELECT AVG(Weight) FROM cars_data WHERE Year = 1978", "difficulty": "easy"},
    {"db_id": "car_1", "question": "What are the names of all continents?", "query": "SELECT Continent FROM continents", "difficulty": "easy"},
    {"db_id": "car_1", "question": "How many models does each car maker have?", "query": "SELECT T1.FullName, count(*) FROM car_makers AS T1 JOIN model_list AS T2 ON T1.Id = T2.Maker GROUP BY T1.Id", "difficulty": "medium"},
    # employee_hire
    {"db_id": "employee_hire", "question": "How many employees are there?", "query": "SELECT count(*) FROM employee", "difficulty": "easy"},
    {"db_id": "employee_hire", "question": "What are the names and cities of all employees?", "query": "SELECT Name, City FROM employee", "difficulty": "easy"},
    {"db_id": "employee_hire", "question": "What is the average age of employees?", "query": "SELECT AVG(Age) FROM employee", "difficulty": "easy"},
    {"db_id": "employee_hire", "question": "How many full-time employees are hired by each shop?", "query": "SELECT Shop_ID, count(*) FROM hiring WHERE Is_full_time = 1 GROUP BY Shop_ID", "difficulty": "medium"},
    {"db_id": "employee_hire", "question": "What is the total bonus awarded in 2020?", "query": "SELECT SUM(Bonus) FROM evaluation WHERE Year_awarded = 2020", "difficulty": "easy"},
    {"db_id": "employee_hire", "question": "What are the names and locations of all shops?", "query": "SELECT Name, Location FROM shop", "difficulty": "easy"},
    {"db_id": "employee_hire", "question": "Show employees who have received a bonus greater than 1000.", "query": "SELECT T1.Name FROM employee AS T1 JOIN evaluation AS T2 ON T1.Employee_ID = T2.Employee_ID WHERE T2.Bonus > 1000", "difficulty": "medium"},
    {"db_id": "employee_hire", "question": "How many shops opened before 2015?", "query": "SELECT count(*) FROM shop WHERE Open_Year < 2015", "difficulty": "easy"},
    {"db_id": "employee_hire", "question": "What is the maximum number of products in any shop?", "query": "SELECT MAX(Number_products) FROM shop", "difficulty": "easy"},
    {"db_id": "employee_hire", "question": "What are the names of employees who work in shops located in Mall A?", "query": "SELECT T2.Name FROM hiring AS T1 JOIN employee AS T2 ON T1.Employee_ID = T2.Employee_ID JOIN shop AS T3 ON T1.Shop_ID = T3.Shop_ID WHERE T3.Location = 'Mall A'", "difficulty": "hard"},
    # student_transcript
    {"db_id": "student_transcript", "question": "How many students are there?", "query": "SELECT count(*) FROM students", "difficulty": "easy"},
    {"db_id": "student_transcript", "question": "What are the names of all courses?", "query": "SELECT course_name FROM courses", "difficulty": "easy"},
    {"db_id": "student_transcript", "question": "How many courses is each student registered for?", "query": "SELECT student_id, count(*) FROM student_course_registrations GROUP BY student_id", "difficulty": "medium"},
    {"db_id": "student_transcript", "question": "What are the first and last names of students from Boston?", "query": "SELECT T1.first_name, T1.last_name FROM students AS T1 JOIN addresses AS T2 ON T1.address_id = T2.address_id WHERE T2.city = 'Boston'", "difficulty": "medium"},
    {"db_id": "student_transcript", "question": "How many students are there per city?", "query": "SELECT T2.city, count(*) FROM students AS T1 JOIN addresses AS T2 ON T1.address_id = T2.address_id GROUP BY T2.city", "difficulty": "medium"},
    {"db_id": "student_transcript", "question": "What courses did student 1 attend?", "query": "SELECT T2.course_name FROM student_course_attendance AS T1 JOIN courses AS T2 ON T1.course_id = T2.course_id WHERE T1.student_id = 1", "difficulty": "easy"},
    {"db_id": "student_transcript", "question": "How many students attended each course?", "query": "SELECT course_id, count(*) FROM student_course_attendance GROUP BY course_id", "difficulty": "easy"},
    {"db_id": "student_transcript", "question": "What is the registration date of student 2 for each course?", "query": "SELECT course_id, registration_date FROM student_course_registrations WHERE student_id = 2", "difficulty": "easy"},
    {"db_id": "student_transcript", "question": "What are the last names of all students?", "query": "SELECT last_name FROM students", "difficulty": "easy"},
    {"db_id": "student_transcript", "question": "Which courses have more than 1 student registered?", "query": "SELECT course_id FROM student_course_registrations GROUP BY course_id HAVING count(*) > 1", "difficulty": "medium"},
    # poker_player
    {"db_id": "poker_player", "question": "How many poker players are there?", "query": "SELECT count(*) FROM poker_player", "difficulty": "easy"},
    {"db_id": "poker_player", "question": "What is the total earnings of all poker players?", "query": "SELECT SUM(Earnings) FROM poker_player", "difficulty": "easy"},
    {"db_id": "poker_player", "question": "What are the names of poker players from the USA?", "query": "SELECT T2.Name FROM poker_player AS T1 JOIN people AS T2 ON T1.People_ID = T2.People_ID WHERE T2.Nationality = 'USA'", "difficulty": "medium"},
    {"db_id": "poker_player", "question": "What is the maximum earnings among all poker players?", "query": "SELECT MAX(Earnings) FROM poker_player", "difficulty": "easy"},
    {"db_id": "poker_player", "question": "What is the average money rank of all players?", "query": "SELECT AVG(Money_Rank) FROM poker_player", "difficulty": "easy"},
    {"db_id": "poker_player", "question": "What are the names and heights of all people?", "query": "SELECT Name, Height FROM people", "difficulty": "easy"},
    {"db_id": "poker_player", "question": "How many players have made the final table more than 5 times?", "query": "SELECT count(*) FROM poker_player WHERE Final_Table_Made > 5", "difficulty": "easy"},
    {"db_id": "poker_player", "question": "Show the name and nationality of the player with the highest earnings.", "query": "SELECT T2.Name, T2.Nationality FROM poker_player AS T1 JOIN people AS T2 ON T1.People_ID = T2.People_ID ORDER BY T1.Earnings DESC LIMIT 1", "difficulty": "medium"},
    {"db_id": "poker_player", "question": "How many players are from each nationality?", "query": "SELECT T2.Nationality, count(*) FROM poker_player AS T1 JOIN people AS T2 ON T1.People_ID = T2.People_ID GROUP BY T2.Nationality", "difficulty": "medium"},
    {"db_id": "poker_player", "question": "What is the best finish achieved by any player?", "query": "SELECT MIN(Best_Finish) FROM poker_player", "difficulty": "easy"},
    {"db_id": "poker_player", "question": "What are the names of people taller than 1.75?", "query": "SELECT Name FROM people WHERE Height > 1.75", "difficulty": "easy"},
]

# ---------------------------------------------------------------------------
# 50 BIRD-style examples across 4 databases
# ---------------------------------------------------------------------------

BIRD_DBS = {
    "california_schools": {
        "tables": ["schools", "satscores", "frpm"],
        "columns": {
            "schools": ["CDSCode", "County", "District", "School", "City", "Zip", "Phone", "StatusType", "Charter"],
            "satscores": ["cds", "rtype", "sname", "dname", "cname", "NumTstTakr", "AvgScrRead", "AvgScrMath", "AvgScrWrite", "NumGE1500"],
            "frpm": ["CDSCode", "County Name", "District Name", "School Name", "Low Grade", "High Grade", "Enrollment (K-12)", "Free Meal Count (K-12)", "Percent (%) Eligible Free (K-12)", "FRPM Count (K-12)"],
        },
        "foreign_keys": [],
        "data": {
            "schools": [
                ("01234560000000", "Alameda", ALAMEDA_CITY_UNIFIED, ALAMEDA_HIGH, "Alameda", "94501", "510-748-4000", "Active", 0),
                ("01234560112607", "Alameda", ALAMEDA_CITY_UNIFIED, ENCINAL_HIGH, "Alameda", "94501", "510-748-4010", "Active", 0),
                ("01234560136713", "Alameda", ALAMEDA_CITY_UNIFIED, ISLAND_HIGH, "Alameda", "94501", "510-748-4000", "Active", 1),
                ("01611190130401", "Alpine", ALPINE_COUNTY_UNIFIED, ALPINE_CREST, "Markleeville", "96120", "530-694-2230", "Active", 0),
                ("01611190000000", "Alpine", ALPINE_COUNTY_UNIFIED, "Woodfords", "Markleeville", "96120", "530-694-2230", "Active", 0),
            ],
            "satscores": [
                ("01234560000000", "S", ALAMEDA_HIGH, ALAMEDA_CITY_UNIFIED, "Alameda", 280, 498, 502, 495, 120),
                ("01234560112607", "S", ENCINAL_HIGH, ALAMEDA_CITY_UNIFIED, "Alameda", 210, 480, 475, 472, 85),
                ("01234560136713", "S", ISLAND_HIGH, ALAMEDA_CITY_UNIFIED, "Alameda", 45, 430, 420, 415, 10),
                ("01611190130401", "S", ALPINE_CREST, ALPINE_COUNTY_UNIFIED, "Alpine", 25, 510, 520, 505, 15),
                ("01611190000000", "S", "Woodfords", ALPINE_COUNTY_UNIFIED, "Alpine", 30, 490, 485, 480, 12),
            ],
            "frpm": [
                ("01234560000000", "Alameda", ALAMEDA_CITY_UNIFIED, ALAMEDA_HIGH, "9", "12", 1800, 200, 0.111, 210),
                ("01234560112607", "Alameda", ALAMEDA_CITY_UNIFIED, ENCINAL_HIGH, "9", "12", 1500, 350, 0.233, 360),
                ("01234560136713", "Alameda", ALAMEDA_CITY_UNIFIED, ISLAND_HIGH, "9", "12", 300, 180, 0.600, 185),
                ("01611190130401", "Alpine", ALPINE_COUNTY_UNIFIED, ALPINE_CREST, "K", "12", 200, 40, 0.200, 42),
                ("01611190000000", "Alpine", ALPINE_COUNTY_UNIFIED, "Woodfords", "K", "12", 120, 30, 0.250, 31),
            ],
        },
    },
    "debit_card_specializing": {
        "tables": ["customers", "gasstations", "transactions_1k", "yearmonth", "products"],
        "columns": {
            "customers": ["CustomerID", "Segment", "Currency"],
            "gasstations": ["GasStationID", "ChainID", "Country", "Segment"],
            "transactions_1k": ["TransactionID", "Date", "CustomerID", "CardID", "GasStationID", "ProductID", "Amount", "Price"],
            "yearmonth": ["CustomerID", "Date", "Consumption"],
            "products": ["ProductID", "Description"],
        },
        "foreign_keys": [
            ("transactions_1k", "CustomerID", "customers", "CustomerID"),
            ("transactions_1k", "GasStationID", "gasstations", "GasStationID"),
            ("transactions_1k", "ProductID", "products", "ProductID"),
        ],
        "data": {
            "customers": [(1, "SME", "CZK"), (2, "LAM", "EUR"), (3, "SME", "EUR"), (4, "KAM", "CZK"), (5, "LAM", "CZK")],
            "gasstations": [(1, 1, "CZE", "Value for money"), (2, 2, "SVK", "Premium"), (3, 1, "CZE", "Value for money")],
            "transactions_1k": [
                (1, "2012-01-01", 1, 101, 1, 1, 10, 150.0),
                (2, "2012-01-02", 2, 102, 2, 2, 5, 80.0),
                (3, "2012-01-03", 1, 101, 1, 1, 8, 120.0),
                (4, "2012-01-04", 3, 103, 3, 3, 20, 300.0),
                (5, "2012-01-05", 2, 102, 2, 2, 3, 45.0),
            ],
            "yearmonth": [(1, "2012-01", 850.0), (2, "2012-01", 400.0), (3, "2012-01", 1200.0)],
            "products": [(1, "Diesel"), (2, "Gasoline"), (3, "CNG")],
        },
    },
    "european_football_2": {
        "tables": ["league", "match", "player", "player_attributes", "team", "team_attributes"],
        "columns": {
            "league": ["id", "country_id", "name"],
            "match": ["id", "country_id", "league_id", "season", "stage", "date", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"],
            "player": ["id", "player_api_id", "player_name", "player_fifa_api_id", "birthday", "height", "weight"],
            "player_attributes": ["id", "player_fifa_api_id", "player_api_id", "date", "overall_rating", "potential", "preferred_foot"],
            "team": ["id", "team_api_id", "team_fifa_api_id", "team_long_name", "team_short_name"],
            "team_attributes": ["id", "team_fifa_api_id", "team_api_id", "date", "buildUpPlaySpeed", "defencePressure"],
        },
        "foreign_keys": [],
        "data": {
            "league": [(1, 1, "Belgium Jupiler League"), (2, 2, "England Premier League"), (3, 3, "France Ligue 1")],
            "match": [
                (1, 1, 1, SEASON_2008_2009, 1, "2008-08-02", 9987, 9993, 1, 0),
                (2, 1, 1, SEASON_2008_2009, 1, "2008-08-09", 9993, 9994, 2, 1),
                (3, 2, 2, SEASON_2008_2009, 1, "2008-08-16", 8455, 8659, 0, 0),
                (4, 2, 2, SEASON_2008_2009, 1, "2008-08-17", 8659, 8472, 3, 2),
                (5, 3, 3, SEASON_2008_2009, 1, "2008-08-09", 9825, 9826, 1, 1),
            ],
            "player": [(1, 505942, "Aaron Appindangoye", 218353, "1992-02-29", 182.88, 187), (2, 155782, "Aaron Cresswell", 189615, "1989-12-15", 170.18, 146), (3, 162549, "Aaron Doran", 186170, "1991-05-13", 170.18, 143)],
            "player_attributes": [(1, 218353, 505942, "2016-02-18", 67, 71, "right"), (2, 189615, 155782, "2016-02-18", 72, 72, "left")],
            "team": [(1, 9987, 673, "KSV Cercle Brugge", "CSB"), (2, 9993, 675, "RSC Anderlecht", "AND"), (3, 8455, 1, "Arsenal", "ARS")],
            "team_attributes": [(1, 673, 9987, "2010-02-22", 60, 50), (2, 675, 9993, "2010-02-22", 55, 45)],
        },
    },
    "toxicology": {
        "tables": ["atom", "bond", "connected", "molecule"],
        "columns": {
            "atom": ["atom_id", "molecule_id", "element"],
            "bond": ["bond_id", "molecule_id", "bond_type"],
            "connected": ["atom_id", "atom_id2", "bond_id"],
            "molecule": ["molecule_id", "label"],
        },
        "foreign_keys": [
            ("atom", "molecule_id", "molecule", "molecule_id"),
            ("bond", "molecule_id", "molecule", "molecule_id"),
            ("connected", "bond_id", "bond", "bond_id"),
        ],
        "data": {
            "molecule": [("TR001", "+"), ("TR002", "-"), ("TR003", "+"), ("TR004", "-"), ("TR005", "+")],
            "atom": [("TR001_1", "TR001", "c"), ("TR001_2", "TR001", "n"), ("TR002_1", "TR002", "cl"), ("TR002_2", "TR002", "c"), ("TR003_1", "TR003", "br")],
            "bond": [("TR001_1_2", "TR001", "-"), ("TR001_2_3", "TR001", "="), ("TR002_1_2", "TR002", "-"), ("TR003_1_2", "TR003", "#")],
            "connected": [("TR001_1", "TR001_2", "TR001_1_2"), ("TR001_2", "TR001_1", "TR001_2_3"), ("TR002_1", "TR002_2", "TR002_1_2")],
        },
    },
}

BIRD_EXAMPLES = [
    # california_schools
    {"db_id": "california_schools", "question": "How many schools are there in Alameda county?", "SQL": "SELECT count(*) FROM schools WHERE County = 'Alameda'", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "What is the average SAT math score across all schools?", "SQL": "SELECT AVG(AvgScrMath) FROM satscores", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "How many charter schools are there?", "SQL": "SELECT count(*) FROM schools WHERE Charter = 1", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "What is the total enrollment for all schools in the frpm table?", "SQL": "SELECT SUM(`Enrollment (K-12)`) FROM frpm", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "What is the highest average reading score among all schools?", "SQL": "SELECT MAX(AvgScrRead) FROM satscores", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "How many schools are in each county?", "SQL": "SELECT County, count(*) FROM schools GROUP BY County", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "What are the names and cities of all active schools?", "SQL": "SELECT School, City FROM schools WHERE StatusType = 'Active'", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "What is the average free meal percentage for schools in Alameda?", "SQL": "SELECT AVG(`Percent (%) Eligible Free (K-12)`) FROM frpm WHERE `County Name` = 'Alameda'", "difficulty": "medium", "evidence": ""},
    {"db_id": "california_schools", "question": "How many schools have more than 100 students eligible for free meals?", "SQL": "SELECT count(*) FROM frpm WHERE `Free Meal Count (K-12)` > 100", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "What is the minimum average math SAT score among all schools?", "SQL": "SELECT MIN(AvgScrMath) FROM satscores", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "How many schools have more than 200 SAT test takers?", "SQL": "SELECT count(*) FROM satscores WHERE NumTstTakr > 200", "difficulty": "easy", "evidence": ""},
    {"db_id": "california_schools", "question": "What is the total number of students scoring above 1500 on the SAT across all schools?", "SQL": "SELECT SUM(NumGE1500) FROM satscores", "difficulty": "easy", "evidence": ""},
    # debit_card_specializing
    {"db_id": "debit_card_specializing", "question": "How many customers are there?", "SQL": "SELECT count(*) FROM customers", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What is the total amount of all transactions?", "SQL": "SELECT SUM(Amount) FROM transactions_1k", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "How many transactions used each product?", "SQL": "SELECT ProductID, count(*) FROM transactions_1k GROUP BY ProductID", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What is the average price per transaction?", "SQL": "SELECT AVG(Price) FROM transactions_1k", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "How many gas stations are in each country?", "SQL": "SELECT Country, count(*) FROM gasstations GROUP BY Country", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What is the maximum consumption among all customers in 2012-01?", "SQL": "SELECT MAX(Consumption) FROM yearmonth WHERE Date = '2012-01'", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "How many customers use EUR currency?", "SQL": "SELECT count(*) FROM customers WHERE Currency = 'EUR'", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What are the descriptions of all products?", "SQL": "SELECT Description FROM products", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What is the total price paid by customer 1?", "SQL": "SELECT SUM(Price) FROM transactions_1k WHERE CustomerID = 1", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "How many transactions took place at each gas station?", "SQL": "SELECT GasStationID, count(*) FROM transactions_1k GROUP BY GasStationID", "difficulty": "easy", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What are the segments of all customers?", "SQL": "SELECT DISTINCT Segment FROM customers", "difficulty": "easy", "evidence": ""},
    # european_football_2
    {"db_id": "european_football_2", "question": "How many leagues are there?", "SQL": "SELECT count(*) FROM league", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "What are the names of all leagues?", "SQL": "SELECT name FROM league", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "How many matches ended in a draw (equal home and away goals)?", "SQL": "SELECT count(*) FROM match WHERE home_team_goal = away_team_goal", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "What is the average overall rating of all players?", "SQL": "SELECT AVG(overall_rating) FROM player_attributes", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "How many matches were played in each season?", "SQL": "SELECT season, count(*) FROM match GROUP BY season", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "What are the long names of all teams?", "SQL": "SELECT team_long_name FROM team", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "What is the maximum height of all players?", "SQL": "SELECT MAX(height) FROM player", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "How many players prefer each foot?", "SQL": "SELECT preferred_foot, count(*) FROM player_attributes GROUP BY preferred_foot", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "What is the total number of home goals scored in league 1?", "SQL": "SELECT SUM(home_team_goal) FROM match WHERE league_id = 1", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "What is the average build up play speed of all teams?", "SQL": "SELECT AVG(buildUpPlaySpeed) FROM team_attributes", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "How many players were born after 1990?", "SQL": "SELECT count(*) FROM player WHERE birthday > '1990-01-01'", "difficulty": "easy", "evidence": ""},
    # toxicology
    {"db_id": "toxicology", "question": "How many molecules are there?", "SQL": "SELECT count(*) FROM molecule", "difficulty": "easy", "evidence": ""},
    {"db_id": "toxicology", "question": "How many carcinogenic molecules are there?", "SQL": "SELECT count(*) FROM molecule WHERE label = '+'", "difficulty": "easy", "evidence": "carcinogenic means label = '+'"},
    {"db_id": "toxicology", "question": "How many atoms are there in each molecule?", "SQL": "SELECT molecule_id, count(*) FROM atom GROUP BY molecule_id", "difficulty": "easy", "evidence": ""},
    {"db_id": "toxicology", "question": "What are all the distinct elements in the atom table?", "SQL": "SELECT DISTINCT element FROM atom", "difficulty": "easy", "evidence": ""},
    {"db_id": "toxicology", "question": "How many bonds of each type exist?", "SQL": "SELECT bond_type, count(*) FROM bond GROUP BY bond_type", "difficulty": "easy", "evidence": ""},
    {"db_id": "toxicology", "question": "How many double bonds are there?", "SQL": "SELECT count(*) FROM bond WHERE bond_type = '='", "difficulty": "easy", "evidence": "double bond means bond_type = '='"},
    {"db_id": "toxicology", "question": "What molecule does atom TR001_1 belong to?", "SQL": "SELECT molecule_id FROM atom WHERE atom_id = 'TR001_1'", "difficulty": "easy", "evidence": ""},
    {"db_id": "toxicology", "question": "How many atoms are in non-carcinogenic molecules?", "SQL": "SELECT count(*) FROM atom AS T1 JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.label = '-'", "difficulty": "medium", "evidence": "non-carcinogenic means label = '-'"},
    {"db_id": "toxicology", "question": "How many connections does each atom have?", "SQL": "SELECT atom_id, count(*) FROM connected GROUP BY atom_id", "difficulty": "easy", "evidence": ""},
    {"db_id": "toxicology", "question": "What is the label of molecule TR003?", "SQL": "SELECT label FROM molecule WHERE molecule_id = 'TR003'", "difficulty": "easy", "evidence": ""},
    # extra 6 to reach 50
    {"db_id": "california_schools", "question": "How many school districts are there in Alpine county?", "SQL": "SELECT count(DISTINCT District) FROM schools WHERE County = 'Alpine'", "difficulty": "medium", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What is the total consumption of all customers in January 2012?", "SQL": "SELECT SUM(Consumption) FROM yearmonth WHERE Date = '2012-01'", "difficulty": "easy", "evidence": ""},
    {"db_id": "european_football_2", "question": "What is the total number of goals scored in all matches?", "SQL": "SELECT SUM(home_team_goal + away_team_goal) FROM match", "difficulty": "easy", "evidence": ""},
    {"db_id": "toxicology", "question": "How many molecules are non-carcinogenic?", "SQL": "SELECT count(*) FROM molecule WHERE label = '-'", "difficulty": "easy", "evidence": "non-carcinogenic means label = '-'"},
    {"db_id": "california_schools", "question": "What is the name of the school with the most SAT test takers?", "SQL": "SELECT sname FROM satscores ORDER BY NumTstTakr DESC LIMIT 1", "difficulty": "medium", "evidence": ""},
    {"db_id": "debit_card_specializing", "question": "What is the number of premium gas stations?", "SQL": "SELECT count(*) FROM gasstations WHERE Segment = 'Premium'", "difficulty": "easy", "evidence": ""},
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _create_sqlite(db_path: Path, db_def: dict) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    for table, cols in db_def["columns"].items():
        col_defs = ", ".join(f'"{c}" TEXT' for c in cols)
        cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})')
        rows = db_def["data"].get(table, [])
        if rows:
            placeholders = ", ".join("?" for _ in cols)
            cur.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)
    conn.commit()
    conn.close()


def build_spider_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # tables.json
    tables = []
    for db_id, db_def in SPIDER_DBS.items():
        db_tables = list(db_def["columns"].keys())
        col_names = [[-1, "*"]]
        col_types = ["text"]
        col_to_table = [-1]
        for ti, (tbl, cols) in enumerate(db_def["columns"].items()):
            for col in cols:
                col_names.append([ti, col])
                col_types.append("text")
                col_to_table.append(ti)
        fks = []
        for (ft, fc, pt, pc) in db_def.get("foreign_keys", []):
            try:
                ft_idx = db_tables.index(ft)
                pt_idx = db_tables.index(pt)
                fc_idx = next(i for i, (ti2, c) in enumerate(col_names) if ti2 == ft_idx and c == fc)
                pc_idx = next(i for i, (ti2, c) in enumerate(col_names) if ti2 == pt_idx and c == pc)
                fks.append([fc_idx, pc_idx])
            except (ValueError, StopIteration):
                pass
        tables.append({
            "db_id": db_id,
            "table_names": db_tables,
            "table_names_original": db_tables,
            "column_names": col_names,
            "column_names_original": col_names,
            "column_types": col_types,
            "column_to_table": col_to_table,
            "foreign_keys": fks,
            "primary_keys": [],
        })
    (root / "tables.json").write_text(json.dumps(tables), encoding="utf-8")
    # dev.json
    dev_examples = [
        {**ex, "question_id": i, "difficulty": ex.get("difficulty", "easy")}
        for i, ex in enumerate(SPIDER_EXAMPLES)
    ]
    (root / "dev.json").write_text(json.dumps(dev_examples), encoding="utf-8")
    # SQLite files
    db_root = root / "database"
    for db_id, db_def in SPIDER_DBS.items():
        db_dir = db_root / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        _create_sqlite(db_dir / f"{db_id}.sqlite", db_def)


def build_bird_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dev_examples = [
        {**ex, "question_id": i}
        for i, ex in enumerate(BIRD_EXAMPLES)
    ]
    (root / "dev.json").write_text(json.dumps(dev_examples), encoding="utf-8")
    db_root = root / "dev_databases"
    for db_id, db_def in BIRD_DBS.items():
        db_dir = db_root / db_id
        db_dir.mkdir(parents=True, exist_ok=True)
        _create_sqlite(db_dir / f"{db_id}.sqlite", db_def)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TEXT2QL_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY or TEXT2QL_API_KEY environment variable.")
        sys.exit(1)

    repo_src = str(Path(__file__).parent / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)

    benchmarks_mod = importlib.import_module("text2ql.benchmarks")
    core_mod = importlib.import_module("text2ql.core")
    dataset_mod = importlib.import_module("text2ql.dataset")
    provider_mod = importlib.import_module("text2ql.providers.openai_compatible")

    load_spider = benchmarks_mod.load_spider
    load_bird = benchmarks_mod.load_bird
    run_benchmark = benchmarks_mod.run_benchmark
    format_report = benchmarks_mod.format_report
    benchmark_config_cls = benchmarks_mod.BenchmarkConfig
    text2ql_cls = core_mod.Text2QL
    dataset_example_cls = dataset_mod.DatasetExample
    openai_provider_cls = provider_mod.OpenAICompatibleProvider

    provider = openai_provider_cls(api_key=api_key, model="gpt-4o-mini")
    service = text2ql_cls(provider=provider)

    with tempfile.TemporaryDirectory() as tmpdir:
        spider_root = Path(tmpdir) / "spider"
        bird_root = Path(tmpdir) / "bird"

        print("Building synthetic Spider dataset (50 examples)...")
        build_spider_dataset(spider_root)
        print("Building synthetic BIRD dataset (50 examples)...")
        build_bird_dataset(bird_root)

        # Load
        spider_examples = load_spider(spider_root, split="dev", limit=50)
        bird_examples = load_bird(bird_root, split="dev", limit=50)

        # Patch context to mode="llm"
        def with_llm_mode(examples: list[Any]) -> list[Any]:
            patched = []
            for ex in examples:
                ctx = dict(ex.context)
                ctx["mode"] = "llm"
                patched.append(dataset_example_cls(
                    text=ex.text,
                    target=ex.target,
                    expected_query=ex.expected_query,
                    schema=ex.schema,
                    mapping=ex.mapping,
                    context=ctx,
                    metadata=ex.metadata,
                ))
            return patched

        spider_examples = with_llm_mode(spider_examples)
        bird_examples = with_llm_mode(bird_examples)

        cfg = benchmark_config_cls(mode="execution", service=service, concurrency=1)

        print(f"\nRunning Spider benchmark ({len(spider_examples)} examples, mode=llm, model=gpt-4o-mini)...")
        spider_report = run_benchmark(spider_examples, config=cfg)
        print(format_report(spider_report, verbose=True))

        print(f"\nRunning BIRD benchmark ({len(bird_examples)} examples, mode=llm, model=gpt-4o-mini)...")
        bird_report = run_benchmark(bird_examples, config=cfg)
        print(format_report(bird_report, verbose=True))

        # Summary
        print("\n" + "=" * 68)
        print("  SUMMARY")
        print("=" * 68)
        for label, report in [("Spider", spider_report), ("BIRD", bird_report)]:
            print(f"  {label:<8}  Exact={report.exact_match_accuracy:.1%}  "
                  f"Structural={report.structural_accuracy:.1%}  "
                  f"Execution={report.execution_accuracy:.1%}  "
                  f"Errors={report.errors}/{report.total}")
        print("=" * 68)


if __name__ == "__main__":
    main()
