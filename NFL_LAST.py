import numpy as np
import pandas as pd


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0]) * 60 + int(txt[1]) + int(txt[2]) / 60
    return ans


def strtofloat(x):
    try:
        return float(x)
    except:
        return -1


def OffensePersonnelSplit(x):
    dic = {'DB': 0, 'DL': 0, 'LB': 0, 'OL': 0, 'QB': 0, 'RB': 0, 'TE': 0, 'WR': 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic


def DefensePersonnelSplit(x):
    dic = {'DB': 0, 'DL': 0, 'LB': 0, 'OL': 0}
    for xx in x.split(","):
        xxs = xx.split(" ")
        dic[xxs[-1]] = int(xxs[-2])
    return dic


def orientation_to_cat(x):
    x = np.clip(x, 0, 360 - 1)
    try:
        return str(int(x / 15))
    except:
        return "nan"


def new_X(x_coordinate, play_direction):
    if play_direction == 'left':
        return 120.0 - x_coordinate
    else:
        return x_coordinate


def new_line(rush_team, field_position, yardline):
    if rush_team == field_position:
        # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage
        return 10.0 + yardline
    else:
        # half the field plus the yards between mitrainield and the line of scrimmage
        return 60.0 + (50 - yardline)


def new_orientation(angle, play_direction):
    if play_direction == 'left':
        new_angle = 360.0 - angle
        if new_angle == 360.0:
            new_angle = 0.0
        return new_angle
    else:
        return angle


def euclidean_distance(x1, y1, x2, y2):
    x_diff = (x1 - x2) ** 2
    y_diff = (y1 - y2) ** 2
    return np.sqrt(x_diff + y_diff)


def back_direction(orientation):
    if orientation > 180.0:
        return 1
    else:
        return 0


def offense_formation(x):  # , '': 308
    o_formation = {'ACE': 88, 'EMPTY': 2552, 'WILDCAT': 7524, 'JUMBO': 20130, 'PISTOL': 57684,
                   'I_FORM': 426624, 'SHOTGUN': 680570, 'SINGLEBACK': 951808}
    for key in o_formation.keys():
        if x == key: return o_formation.get(key)


def player_collegeName(x):
    p_collegeName = {'Abilene Christian': 1848, 'Air Force': 1087, 'Akron': 1939, 'Alabama': 67830,
                     'Alabama A&M': 519, 'Alabama State': 1549, 'Alabama-Birmingham': 3683, 'Albany': 180,
                     'Albany State, Ga.': 1168, 'Appalachian State': 4845, 'Arizona': 10092, 'Arizona State': 11238,
                     'Arkansas': 17925, 'Arkansas State': 2922, 'Arkansas-Monticello': 25,
                     'Arkansas-Pine Bluff': 2017, 'Army': 2843, 'Ashland': 1198, 'Assumption': 692, 'Auburn': 22254,
                     'Augustana S.D.': 536, 'Ball State': 1967, 'Baylor': 10633, 'Belhaven': 698, 'Beloit': 847,
                     'Bethune-Cookman': 961, 'Bloomsburg': 2136, 'Boise State': 20461, 'Boston College': 18415,
                     'Bowie State': 373, 'Bowling Green State': 786, 'Brigham Young': 9681, 'Brown': 1873,
                     'Bucknell': 1821, 'Buffalo': 2546, 'California': 32240, 'California-Irvine': 1794,
                     'California Pa.': 1518, 'Canisius': 1267, 'Central Arkansas': 660, 'Central Florida': 18522,
                     'Central Michigan': 14264, 'Central Missouri': 928, 'Chadron State': 56, 'Chattanooga': 2885,
                     'Cincinnati': 13424, 'Citadel': 237, 'Clemson': 37532, 'Coastal Carolina': 3546,
                     'Colorado': 15888, 'Colorado State': 7405, 'Colorado State-Pueblo': 4756, 'Columbia': 738,
                     'Concordia-St. Paul': 649, 'Connecticut': 9651, 'Cornell': 2791, 'Delaware': 9137,
                     'Delaware State': 1869, 'Drake': 454, 'Duke': 10494, 'East Carolina': 5961,
                     'East Central': 1200, 'Eastern Illinois': 1618, 'Eastern Kentucky': 524,
                     'Eastern Michigan': 2334, 'Eastern Washington': 5103, 'Ferris State': 97, 'Florida': 49239,
                     'Florida Atlantic': 6486, 'Florida International': 4803, 'Florida State': 43560,
                     'Fordham': 188, 'Fort Hays State': 674, 'Fresno State': 11176, 'Frostburg State': 151,
                     'Furman': 573, 'Georgia': 49410, 'Georgia Southern': 2238, 'Georgia State': 1290,
                     'Georgia Tech': 13598, 'Grambling': 1599, 'Grambling State': 571, 'Grand Valley State': 4045,
                     'Greenville': 1500, 'Hampton': 1269, 'Harvard': 5216, 'Hawaii': 113, 'Henderson State': 400,
                     'Hillsdale': 3020, 'Hobart': 1832, 'Houston': 11104, 'Howard': 2938, 'Humboldt State': 78,
                     'Idaho': 5793, 'Idaho State': 3545, 'Illinois': 14013, 'Illinois State': 2753,
                     'Indiana': 12360, 'Indiana State': 67, 'Iowa': 31158, 'Iowa State': 5993, 'Jacksonville': 409,
                     'Jacksonville State': 208, 'James Madison': 2338, 'Kansas': 7510, 'Kansas State': 12718,
                     'Kent State': 8277, 'Kentucky': 16943, 'Kentucky Wesleyan': 1623, 'Lamar': 167,
                     'Laval, Can.': 794, 'Liberty': 26, 'Limestone': 119, 'Lindenwood': 1572,
                     'Louisiana Coll.': 1580, 'Louisiana State': 51863, 'Louisiana Tech': 12535,
                     'Louisiana-Lafayette': 1449, 'Louisville': 21024, 'LSU': 13320, 'Maine': 715,
                     'Manitoba, Can.': 1529, 'Marist': 301, 'Mars Hill': 795, 'Marshall': 4537,
                     'Mary Hardin-Baylor': 46, 'Maryland': 14538, 'Massachusetts': 3415, 'McGill': 1513,
                     'Memphis': 10006, 'Miami': 27346, 'Miami (Fla.)': 6522, 'Miami (Ohio)': 2944,
                     'Miami, O.': 3199, 'Michigan': 34073, 'Michigan State': 21725, 'Michigan Tech': 1803,
                     'Middle Tennessee': 3316, 'Middle Tennessee State': 878, 'Midwestern State': 595,
                     'Minn. State-Mankato': 3483, 'Minnesota': 8909, 'Mississippi': 24130,
                     'Mississippi State': 23543, 'Missouri': 19673, 'Missouri Southern': 1485,
                     'Missouri Southern State': 751, 'Missouri State': 437, 'Missouri Western State': 851,
                     'Monmouth (N.J.)': 2013, 'Monmouth, N.J.': 308, 'Montana': 3546, 'Montana State': 2178,
                     'Mount Union': 1321, 'nan': 170, 'Navy': 22, 'Nebraska': 20372, 'Nebraska-Omaha': 476,
                     'Nevada': 8703, 'Nevada-Las Vegas': 1079, 'New Hampshire': 1005, 'New Mexico': 4134,
                     'New Mexico State': 1199, 'Newberry': 3126, 'Nicholls State': 543, 'No College': 340,
                     'North Alabama': 2459, 'North Carolina': 25334, 'North Carolina A&T': 2470,
                     'North Carolina Central': 1442, 'North Carolina State': 22615,
                     'North Carolina-Charlotte': 1892, 'North Dakota State': 7877, 'North Greenville': 112,
                     'North Texas': 1866, 'Northeast Mississippi CC': 464, 'Northern Illinois': 5100,
                     'Northern Iowa': 4507, 'Northwest Missouri State': 1377, 'Northwestern': 6675,
                     'Northwestern (Ia)': 222, 'Northwestern State-Louisiana': 639, 'Notre Dame': 41454,
                     'Ohio': 3577, 'Ohio State': 59466, 'Ohio U.': 1478, 'Oklahoma': 35433, 'Oklahoma State': 14549,
                     'Old Dominion': 1129, 'Oregon': 29294, 'Oregon State': 16581, 'Ouachita Baptist': 476,
                     'Penn State': 31989, 'Pennsylvania': 2522, 'Pittsburg State': 1570, 'Pittsburgh': 19305,
                     'Portland State': 2770, 'Prairie View': 194, 'Presbyterian': 405, 'Princeton': 983,
                     'Purdue': 15057, 'Regina, Can.': 3750, 'Rice': 6590, 'Richmond': 765, 'Rutgers': 17010,
                     'Sacramento State': 2509, 'Saginaw Valley State': 2406, 'Sam Houston State': 2136,
                     'Samford': 5761, 'San Diego': 585, 'San Diego State': 3649, 'San Jose State': 5729,
                     'Shepherd': 157, 'Shippensburg': 2652, 'Slippery Rock': 1963, 'South Alabama': 649,
                     'South Carolina': 31339, 'South Carolina State': 4016, 'South Dakota': 1832,
                     'South Dakota State': 2891, 'South Florida': 10516, 'Southeast Missouri': 293,
                     'Southeastern Louisiana': 2854, 'Southern Arkansas': 843, 'Southern California': 34264,
                     'Southern Connecticut State': 139, 'Southern Illinois': 929, 'Southern Methodist': 11944,
                     'Southern Mississippi': 12226, 'Southern University': 98, 'Southern Utah': 1242,
                     'St. Francis (PA)': 124, 'Stanford': 35808, 'Stephen F. Austin St.': 328, 'Stillman': 613,
                     'Stony Brook': 168, 'Syracuse': 5135, 'Temple': 15859, 'Tennessee': 20469,
                     'Tennessee State': 1387, 'Tennessee-Chattanooga': 568, 'Texas': 26968, 'Texas A&M': 25617,
                     'Texas Christian': 14639, 'Texas State': 1677, 'Texas Tech': 9860, 'Texas-El Paso': 5046,
                     'Texas-San Antonio': 2411, 'Tiffin University': 805, 'Toledo': 8576, 'Towson': 2035,
                     'Troy': 2980, 'Tulane': 3522, 'Tulsa': 2015, 'UCLA': 30040, 'USC': 18262, 'Utah': 21833,
                     'Utah State': 14047, 'Valdosta State': 4236, 'Vanderbilt': 14889, 'Villanova': 755,
                     'Virginia': 11528, 'Virginia Commonwealth': 544, 'Virginia State': 192, 'Virginia Tech': 16402,
                     'Wagner': 165, 'Wake Forest': 7431, 'Washburn': 740, 'Washington': 26828,
                     'Washington State': 4208, 'Weber State': 523, 'West Alabama': 6565, 'West Georgia': 1323,
                     'West Texas A&M': 1088, 'West Virginia': 16823, 'Western Carolina': 79,
                     'Western Kentucky': 7077, 'Western Michigan': 4806, 'Western Oregon': 2464,
                     'Western State, Colo.': 903, 'William & Mary': 1744, 'William Penn': 2112,
                     'Winston-Salem State': 519, 'Wis.-Platteville': 100, 'Wisconsin': 35247,
                     'Wisconsin-Milwaukee': 1519, 'Wisconsin-Whitewater': 107, 'Wofford': 209, 'Wyoming': 8964,
                     'Yale': 937, 'Youngstown State': 498}
    for key in p_collegeName.keys():
        if x == key: return p_collegeName.get(key)


def position(x):
    p = {'C': 101478, 'CB': 246361, 'DB': 8968, 'DE': 151273, 'DL': 107, 'DT': 147279, 'FB': 12543, 'FS': 103235,
         'G': 176303, 'HB': 2797, 'ILB': 75371, 'LB': 29240, 'MLB': 43170, 'NT': 33568, 'OG': 12515, 'OLB': 140599,
         'OT': 25806, 'QB': 98197, 'RB': 100951, 'S': 7437, 'SAF': 79, 'SS': 86365, 'T': 176808, 'TE': 134522,
         'WR': 232316}
    for key in p.keys():
        if x == key: return p.get(key)


def team_abbr(x):
    t_abbr = {'CAR': 62722, 'PHI': 57618, 'NO': 83864, 'PIT': 63184, 'MIN': 69454, 'CHI': 64394,
              'HST': 64526, 'DET': 71962, 'JAX': 67408, 'IND': 64812, 'TB': 51084, 'ARZ': 65120, 'TEN': 66044,
              'LAC': 74888, 'GB': 58190, 'KC': 67760, 'WAS': 58300, 'NE': 83952, 'CIN': 59158, 'SEA': 70510,
              'ATL': 68398, 'DAL': 74228, 'BLT': 76736, 'DEN': 74558, 'BUF': 61226, 'MIA': 61644, 'CLV': 61402,
              'OAK': 66000, 'SF': 72578, 'NYJ': 60918, 'NYG': 68728, 'LA': 86262, 'ARI': 12496, 'CLE': 4158,
              'HOU': 3608}
    for key in t_abbr.keys():
        if x == key: return t_abbr.get(key)


def stadium(x):
    s = {'Arrowhead Stadium': 68486, 'AT&T Stadium': 66330, 'Bank of America Stadium': 66000,
         'Broncos Stadium At Mile High': 38082, 'CenturyLink Field': 75636, 'Estadio Azteca': 4356,
         'EverBank Field': 31438, 'FedExField': 61248, 'First Energy Stadium': 61160, 'Ford Field': 59158,
         'Gillette Stadium': 80476, 'Hard Rock Stadium': 60324, 'Heinz Field': 63514, 'Lambeau Field': 64922,
         'Levis Stadium': 74338, 'Lincoln Financial Field': 60940, 'Los Angeles Memorial Coliseum': 67958,
         'Lucas Oil Stadium': 65648, 'M&T Bank Stadium': 72446, 'Mercedes-Benz Stadium': 60962,
         'Mercedes-Benz Superdome': 69058, 'MetLife Stadium': 147576, 'New Era Field': 70422,
         'Nissan Stadium': 66286, 'NRG Stadium': 62700, 'Oakland-Alameda County Coliseum': 65032,
         'Paul Brown Stadium': 74844, 'Raymond James Stadium': 57530, 'Soldier Field': 57420,
         'Sports Authority Field at Mile High': 30228, 'State Farm Stadium': 38016, 'StubHub Center': 64680,
         'TIAA Bank Field': 22264, 'Twickenham Stadium': 7590, 'U.S. Bank Stadium': 65582,
         'University of Phoenix Stadium': 25894, 'Wembley Stadium': 18744}
    for key in s.keys():
        if x == key: return s.get(key)


def location(x):
    l = {'Arlington, Texas': 2728, 'Arlington, TX': 2728, 'Atlanta, GA': 65010, 'Baltimore, Maryland': 4180,
         'Baltimore, Md.': 4180, 'Carson, CA': 64680, 'Charlotte, NC': 26334, 'Charlotte, North Carolina': 26334,
         'Chicago, IL': 3828, 'Chicago. IL': 3828, 'Chicago': 3828, 'Cincinnati, OH': 8668,
         'Cincinnati, Ohio': 8668,
         'Cleveland': 2024, 'Cleveland Ohio': 2024, 'Cleveland, OH': 2024, 'Cleveland, Ohio': 2024,
         'Cleveland,Ohio': 2024, 'Denver, CO': 68310, 'Detroit': 20020, 'Detroit, MI': 20020,
         'E. Rutherford, NJ': 8646, 'East Rutherford, N.J.': 8646, 'East Rutherford, NJ': 8646,
         'Foxborough, Ma': 8822, 'Foxborough, MA': 8822, 'Glendale, AZ': 63910, 'Green Bay, WI': 64922,
         'Houston, Texas': 25432, 'Houston, TX': 25432, 'Indianapolis, Ind.': 65648, 'Jacksonville Florida': 2574,
         'Jacksonville, Fl': 2574, 'Jacksonville, FL': 2574, 'Jacksonville, Florida': 2574,
         'Kansas City,  MO': 4994, 'Kansas City, MO': 4994, 'Landover, MD': 61248, 'London': 3674,
         'London, England': 3674, 'Los Angeles, CA': 3894, 'Los Angeles, Calif.': 3894, 'Mexico City': 4356,
         'Miami Gardens, FLA': 7568, 'Miami Gardens, Fla.': 7568, 'Minneapolis, MN': 65582, 'Nashville, TN': 66286,
         'New Orleans': 5566, 'New Orleans, LA': 5566, 'New Orleans, La.': 5566, 'Oakland, CA': 65032,
         'Orchard Park NY': 5610, 'Orchard Park, NY': 5610, 'Philadelphia, PA': 5258, 'Philadelphia, Pa.': 5258,
         'Pittsburgh': 7898, 'Pittsburgh, PA': 7898, 'Santa Clara, CA': 74338, 'Seattle': 13266,
         'Seattle, WA': 13266, 'Tampa, FL': 57530}
    for key in l.keys():
        if x == key: return l.get(key)


def stadium_type(x):  # '': 146652,
    s_type = {'Bowl': 3608, 'Closed Dome': 6446, 'Cloudy': 1694, 'Dome': 6446, 'Dome, closed': 6446,
              'Domed': 6446, 'Domed, closed': 6446, 'Domed, Open': 3388, 'Domed, open': 3388, 'Heinz Field': 4532,
              'Indoor': 62414, 'Indoor, Open Roof': 3388, 'Indoor, Roof Closed': 6446, 'Indoors': 62414,
              'Open': 38786, 'Oudoor': 2794, 'Ourdoor': 2794, 'Outddors': 2794, 'Outdoor': 2794,
              'Outdoor Retr Roof-Open': 2112, 'Outdoors': 2794, 'Outdor': 2794, 'Outside': 2750,
              'Retr. Roof - Closed': 5324, 'Retr. Roof - Open': 2112, 'Retr. Roof Closed': 5324,
              'Retr. Roof-Closed': 5324, 'Retr. Roof-Open': 2112, 'Retractable Roof': 2112}
    for key in s_type.keys():
        if x == key: return s_type.get(key)


def turf(x):
    t = {'Grass': 7040, 'Natural Grass': 7040, 'Field Turf': 13002, 'Artificial': 13002, 'FieldTurf': 13002,
         'UBU Speed Series-S5-M': 36234, 'A-Turf Titan': 70422, 'Twenty-Four/Seven Turf': 13002,
         'UBU Sports Speed S5-M': 36234, 'FieldTurf360': 13002, 'SISGrass': 33484, 'DD GrassMaster': 31438,
         'FieldTurf 360': 13002, 'Natural grass': 7040, 'Artifical': 13002, 'Natural': 7040, 'Field turf': 13002,
         'Naturall Grass': 7040, 'natural grass': 7040, 'grass': 7040}
    for key in t.keys():
        if x == key: return t.get(key)


def game_weather(x):  # '': 176418,
    g_weather = {'30% Chance of Rain': 6226, 'Clear': 7876, 'Clear and cold': 5126,
                 'Clear and Cool': 5126, 'Clear and sunny': 5060, 'Clear and Sunny': 5060, 'Clear and warm': 6358,
                 'Clear skies': 7876, 'Clear Skies': 7876, 'Cloudy': 6270, 'cloudy': 6270,
                 'Cloudy and cold': 3828, 'Cloudy and Cool': 3828,
                 'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.': 2706,
                 'Cloudy, 50% change of rain': 4378, 'Cloudy, chance of rain': 4378,
                 'Cloudy, fog started developing in 2nd quarter': 5280,
                 '"Cloudy, light snow accumulating 1-3"""': 2244, 'Cloudy, Rain': 5104, 'Cold': 3520,
                 'Controlled Climate': 48884, 'Coudy': 6270, 'Fair': 7876, 'Hazy': 13596,
                 'Heavy lake effect snow': 7546, 'Indoor': 22374, 'Indoors': 22374, 'Light Rain': 11880,
                 'Mostly cloudy': 6270, 'Mostly Cloudy': 6270, 'Mostly Coudy': 6270, 'Mostly sunny': 4312,
                 'Mostly Sunny': 4312, 'Mostly Sunny Skies': 4312, 'N/A (Indoors)': 223, 'N/A Indoor': 223,
                 'Overcast': 8360, 'Partly clear': 2640, 'Partly Cloudy': 4378, 'Partly cloudy': 4378,
                 'Partly Clouidy': 4378, 'Partly sunny': 7458, 'Partly Sunny': 7458, 'Party Cloudy': 4378,
                 'Rain': 4532, 'Rain Chance 40%': 4378, 'Rain likely, temps in low 40s.': 4378,
                 'Rain shower': 4554, 'Rainy': 4532, 'Scattered Showers': 4554, 'Showers': 4554, 'Snow': 7986,
                 'Sun & clouds': 3982, 'Sunny': 5060, 'Sunny and clear': 5060, 'Sunny and cold': 4444,
                 'Sunny and warm': 4994, 'Sunny Skies': 4312, 'Sunny, highs to upper 80s': 4312,
                 'Sunny, Windy': 3542, 'T: 51; H: 55; W: NW 10 mph': 3894}
    for key in g_weather.keys():
        if x == key: return g_weather.get(key)


def wind_direction(x):  # '': 325534,
    w_direction = {'1': 6270, '8': 6666, '13': 2992, 'Calm': 4026, 'E': 3520, 'East': 3520,
                   'EAST': 3520, 'East North East': 3762, 'East Southeast': 4950, 'ENE': 3762, 'ESE': 4950,
                   'From ESE': 4950, 'From NNE': 4862, 'From NNW': 3388, 'From S': 3520, 'From SSE': 4004,
                   'From SSW': 2068, 'From SW': 4664, 'From W': 2310, 'from W': 2310, 'From WSW': 4444,
                   'N': 66770, 'N-NE': 4862, 'NE': 6842, 'NNE': 4862, 'NNW': 3388, 'North': 66770,
                   'North East': 6842, 'North/Northwest': 3388, 'NorthEast': 6842, 'Northeast': 6842,
                   'Northwest': 16720, 'NW': 16720, 's': 3520, 'S': 3520, 'SE': 6402, 'South': 3520,
                   'South Southeast': 4004, 'South Southwest': 2068, 'Southeast': 6402, 'SouthWest': 4664,
                   'Southwest': 4664, 'SSE': 4004, 'SSW': 2068, 'SW': 4664, 'W': 2310, 'W-NW': 4026,
                   'W-SW': 4444, 'West': 2310, 'West Northwest': 4026, 'West-Southwest': 4444, 'WNW': 4026,
                   'WSW': 4444}
    for key in w_direction.keys():
        if x == key: return w_direction.get(key)


def wind_speed(x):
    if (x == 'From SSW' or x == 'SSW'): x = ''  # '': 338426,
    w_speed = {'0': 57992, '1': 51788, '2': 99924, '3': 112310, '4': 143000, '5': 231550, '6': 187154,
               '7': 171094, '8': 130878, '9': 129338, '10': 127402, '11': 68354, '12': 100188, '13': 37312,
               '14': 23716, '15': 55308, '16': 39578, '17': 17094, '18': 6600, '19': 2486, '20': 4862, '22': 2398,
               '23': 4664, '24': 3872}
    for key in w_speed.keys():
        if x == key: return w_speed.get(key)


def offense_personnel(x):
    o_personal = {'0 RB, 0 TE, 5 WR': 66, '0 RB, 1 TE, 4 WR': 6600, '0 RB, 2 TE, 3 WR': 352,
                  '0 RB, 3 TE, 2 WR': -44, '1 RB, 0 TE, 3 WR,1 DB': 814, '1 RB, 0 TE, 4 WR': 10076,
                  '1 RB, 1 TE, 2 WR,1 DB': 1936, '1 RB, 1 TE, 2 WR,1 DL': 2178, '1 RB, 1 TE, 2 WR,1 LB': 198,
                  '1 RB, 1 TE, 3 WR': 1061852, '1 RB, 2 TE, 1 WR,1 DB': 308, '1 RB, 2 TE, 1 WR,1 DL': 12298,
                  '1 RB, 2 TE, 1 WR,1 LB': 1122, '1 RB, 2 TE, 2 WR': 454410, '1 RB, 2 TE, 3 WR': 44,
                  '1 RB, 3 TE, 0 WR,1 DB': 110, '1 RB, 3 TE, 0 WR,1 DL': 286, '1 RB, 3 TE, 0 WR,1 LB': 110,
                  '1 RB, 3 TE, 1 WR': 111584, '1 RB, 4 TE, 0 WR': 0, '2 QB, 1 RB, 0 TE, 3 WR': 638,
                  '2 QB, 1 RB, 1 TE, 2 WR': 6820, '2 QB, 1 RB, 2 TE, 1 WR': 836, '2 QB, 1 RB, 3 TE, 0 WR': 220,
                  '2 QB, 2 RB, 0 TE, 2 WR': 176, '2 QB, 2 RB, 1 TE, 1 WR': 4290, '2 QB, 2 RB, 2 TE, 0 WR': 132,
                  '2 QB, 3 RB, 1 TE, 0 WR': 44, '2 RB, 0 TE, 3 WR': 14784, '2 RB, 1 TE, 1 WR,1 DB': 66,
                  '2 RB, 1 TE, 2 WR': 233794, '2 RB, 2 TE, 0 WR,1 DL': -44, '2 RB, 2 TE, 1 WR': 105908,
                  '2 RB, 3 TE, 0 WR': 1144, '2 RB, 3 TE, 1 WR': 66, '3 RB, 0 TE, 2 WR': 726,
                  '3 RB, 1 TE, 1 WR': 2244, '3 RB, 2 TE, 0 WR': 506, '6 OL, 0 RB, 2 TE, 2 WR': -22,
                  '6 OL, 1 RB, 0 TE, 3 WR': 9218, '6 OL, 1 RB, 1 TE, 0 WR,2 DL': 22,
                  '6 OL, 1 RB, 1 TE, 1 WR,1 DL': 440, '6 OL, 1 RB, 1 TE, 2 WR': 27874,
                  '6 OL, 1 RB, 2 TE, 0 WR,1 DL': 154, '6 OL, 1 RB, 2 TE, 0 WR,1 LB': 308,
                  '6 OL, 1 RB, 2 TE, 1 WR': 35156, '6 OL, 1 RB, 3 TE, 0 WR': 1496, '6 OL, 2 RB, 0 TE, 2 WR': 5654,
                  '6 OL, 2 RB, 1 TE, 0 WR,1 DL': 66, '6 OL, 2 RB, 1 TE, 1 WR': 22990,
                  '6 OL, 2 RB, 2 TE, 0 WR': 4400, '6 OL, 3 RB, 0 TE, 1 WR': 110, '7 OL, 1 RB, 0 TE, 2 WR': 1980,
                  '7 OL, 1 RB, 2 TE, 0 WR': 88, '7 OL, 2 RB, 0 TE, 1 WR': 726, '7 OL, 2 RB, 1 TE, 0 WR': -22}
    for key in o_personal.keys():
        if x == key: return o_personal.get(key)


def defense_personnel(x):
    d_personal = {'0 DL, 4 LB, 7 DB': 704, '0 DL, 5 LB, 6 DB': 550, '0 DL, 6 LB, 5 DB': 176,
                  '1 DL, 2 LB, 8 DB': 242, '1 DL, 3 LB, 7 DB': 1914, '1 DL, 4 LB, 6 DB': 7304,
                  '1 DL, 5 LB, 5 DB': 3036, '2 DL, 2 LB, 7 DB': 2706, '2 DL, 3 LB, 6 DB': 57926,
                  '2 DL, 4 LB, 4 DB, 1 OL': 418, '2 DL, 4 LB, 5 DB': 251262, '2 DL, 5 LB, 4 DB': 1584,
                  '3 DL, 1 LB, 7 DB': 1166, '3 DL, 2 LB, 6 DB': 22594, '3 DL, 3 LB, 5 DB': 220418,
                  '3 DL, 4 LB, 3 DB, 1 OL': 66, '3 DL, 4 LB, 4 DB': 313720, '3 DL, 5 LB, 3 DB': 5148,
                  '4 DL, 0 LB, 7 DB': 198, '4 DL, 1 LB, 6 DB': 47806, '4 DL, 2 LB, 5 DB': 636086,
                  '4 DL, 3 LB, 4 DB': 537658, '4 DL, 4 LB, 3 DB': 10208, '4 DL, 5 LB, 1 DB, 1 OL': 22,
                  '4 DL, 5 LB, 2 DB': 132, '4 DL, 6 LB, 1 DB': 44, '5 DL, 1 LB, 5 DB': 2310,
                  '5 DL, 2 LB, 4 DB': 16280, '5 DL, 3 LB, 2 DB, 1 OL': 66, '5 DL, 3 LB, 3 DB': 3410,
                  '5 DL, 4 LB, 1 DB, 1 OL': 44, '5 DL, 4 LB, 2 DB': 990, '5 DL, 5 LB, 1 DB': 110,
                  '6 DL, 1 LB, 4 DB': 0, '6 DL, 2 LB, 3 DB': -44, '6 DL, 3 LB, 2 DB': 572, '6 DL, 4 LB, 1 DB': 440,
                  '7 DL, 2 LB, 2 DB': 22, "0 DL, 4 LB, 6 DB, 1 RB": 0, "1 DL, 3 LB, 6 DB, 1 RB": 0,
                  "1 DL, 4 LB, 5 DB, 1 RB": 0, "2 DL, 3 LB, 5 DB, 1 RB": 0, "2 DL, 4 LB, 4 DB, 1 RB": 0,
                  "3 DL, 4 LB, 3 DB, 1 RB": 0}
    for key in d_personal.keys():
        if x == key: return d_personal.get(key)


def team(x):
    t = {'home': 0, 'away': 1}
    for key in t.keys():
        if x == key: return t.get(key)


def play_direction(x):
    p_direction = {'left': 0, 'right': 1}
    for key in p_direction.keys():
        if x == key: return p_direction.get(key)



def preprocess(train):

    ## Orientation and Dir
    train["Orientation"] = train["Orientation"].fillna(7.272)
    train["Orientation_sin"] = train["Orientation"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
    train["Orientation_cos"] = train["Orientation"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))

    train["Dir"] = train["Dir"].fillna(7.272)
    train["Dir_sin"] = train["Dir"].apply(lambda x: np.sin(x / 360 * 2 * np.pi))
    train["Dir_cos"] = train["Dir"].apply(lambda x: np.cos(x / 360 * 2 * np.pi))


    # updating yardline
    new_yardline = train[train['NflId'] == train['NflIdRusher']]
    new_yardline['YardLine_up'] = new_yardline[['PossessionTeam', 'FieldPosition', 'YardLine']].apply(
        lambda x: new_line(x[0], x[1], x[2]), axis=1)
    new_yardline = new_yardline[['GameId', 'PlayId', 'YardLine_up']]

    # updating orientation
    train['X_up'] = train[['X', 'PlayDirection']].apply(lambda x: new_X(x[0], x[1]), axis=1)
    train['Orientation_up'] = train[['Orientation', 'PlayDirection']].apply(lambda x: new_orientation(x[0], x[1]),
                                                                            axis=1)
    train['Dir_up'] = train[['Dir', 'PlayDirection']].apply(lambda x: new_orientation(x[0], x[1]), axis=1)

    # df = df.drop('YardLine', axis=1)
    up_ort = pd.merge(train, new_yardline, on=['GameId', 'PlayId'], how='inner')

    # updating back features
    carriers = up_ort[up_ort['NflId'] == up_ort['NflIdRusher']][
        ['GameId', 'PlayId', 'NflIdRusher', 'X_up', 'Y', 'Orientation_up', 'Dir_up', 'YardLine_up']]
    carriers['back_from_scrimmage'] = carriers['YardLine_up'] - carriers['X_up']
    carriers['back_oriented_down_field'] = carriers['Orientation_up'].apply(lambda x: back_direction(x))
    carriers['back_moving_down_field'] = carriers['Dir_up'].apply(lambda x: back_direction(x))

    carriers = carriers.rename(columns={'X_up': 'back_X', 'Y': 'back_Y'})
    carriers = carriers[
        ['GameId', 'PlayId', 'NflIdRusher', 'back_X', 'back_Y', 'back_from_scrimmage', 'back_oriented_down_field',
         'back_moving_down_field']]

    # features_relative_to_back

    player_distance = up_ort[['GameId', 'PlayId', 'NflId', 'X_up', 'Y']]
    player_distance = pd.merge(player_distance, carriers, on=['GameId', 'PlayId'], how='inner')
    player_distance = player_distance[player_distance['NflId'] != player_distance['NflIdRusher']]
    player_distance['dist_to_back'] = player_distance[['X_up', 'Y', 'back_X', 'back_Y']].apply(
        lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

    player_distance = player_distance.groupby(
        ['GameId', 'PlayId', 'back_from_scrimmage', 'back_oriented_down_field', 'back_moving_down_field']).agg(
        {'dist_to_back': ['min', 'max', 'mean', 'std']}).reset_index()
    player_distance.columns = ['GameId', 'PlayId', 'back_from_scrimmage', 'back_oriented_down_field',
                               'back_moving_down_field',
                               'min_dist', 'max_dist', 'mean_dist', 'std_dist']

    # defense_features
    rusher = up_ort[up_ort['NflId'] == up_ort['NflIdRusher']][['GameId', 'PlayId', 'Team', 'X_up', 'Y']]
    rusher.columns = ['GameId', 'PlayId', 'RusherTeam', 'RusherX', 'RusherY']

    defense = pd.merge(train, rusher, on=['GameId', 'PlayId'], how='inner')
    defense = defense[defense['Team'] != defense['RusherTeam']][
        ['GameId', 'PlayId', 'X', 'Y', 'RusherX', 'RusherY']]
    defense['def_dist_to_back'] = defense[['X', 'Y', 'RusherX', 'RusherY']].apply(
        lambda x: euclidean_distance(x[0], x[1], x[2], x[3]), axis=1)

    defense = defense.groupby(['GameId', 'PlayId']).agg(
        {'def_dist_to_back': ['min', 'max', 'mean', 'std']}).reset_index()
    defense.columns = ['GameId', 'PlayId', 'def_min_dist', 'def_max_dist', 'def_mean_dist', 'def_std_dist']

    # merging
    temp = pd.merge(player_distance, defense, on=['GameId', 'PlayId'], how='inner')
    train = pd.merge(train, temp, on=['GameId', 'PlayId'], how='inner')

    ## GameClock
    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)
    train["GameClock_minute"] = train["GameClock"].apply(lambda x: x.split(":")[0])

    ## Height
    train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12 * int(x.split('-')[0]) + int(x.split('-')[1]))

    ## Time
    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    ## Age
    seconds_in_year = 60 * 60 * 24 * 365.25
    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    train['PlayerAge'] = train.apply(
        lambda row: (row['TimeHandoff'] - row['PlayerBirthDate']).total_seconds() / seconds_in_year, axis=1)

    ## OffensePersonnel
    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(
        lambda x: pd.Series(OffensePersonnelSplit(x)))
    temp.columns = ["Offense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on="PlayId")

    ## DefensePersonnel
    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(
        lambda x: pd.Series(DefensePersonnelSplit(x)))
    temp.columns = ["Defense" + c for c in temp.columns]
    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]
    train = train.merge(temp, on="PlayId")


    ## diff Score
    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]

    # Rusher
    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])
    temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team": "RusherTeam"})
    train = train.merge(temp, on="PlayId")
    train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]
    train['IsRusher'] = train['IsRusher'].apply(lambda x: 1 if x is True else 0)
    train["IsRusherTeam"] = train["IsRusherTeam"].apply(lambda x: 1 if x is True else 0)
    train["RusherTeam"] = train["RusherTeam"].apply(lambda x: 1 if x is 'away' else 0)


    train['Team'] = train['Team'].apply(lambda x: team(x))
    train['PlayDirection'] = train['PlayDirection'].apply(lambda x: play_direction(x))

    train['OffenseFormation'] = train['OffenseFormation'].apply(lambda x: offense_formation(x))
    train['OffenseFormation'] = train['OffenseFormation'].fillna(308)

    train['PlayerCollegeName'] = train['PlayerCollegeName'].apply(lambda x: player_collegeName(x))
    train['Position'] = train['Position'].apply(lambda x: position(x))
    train['HomeTeamAbbr'] = train['HomeTeamAbbr'].apply(lambda x: team_abbr(x))
    train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].apply(lambda x: team_abbr(x))
    train['Stadium'] = train['Stadium'].apply(lambda x: stadium(x))
    train['Location'] = train['Location'].apply(lambda x: location(x))

    train['StadiumType'] = train['StadiumType'].apply(lambda x: stadium_type(x))
    train['StadiumType'] = train['StadiumType'].fillna(146652)


    train['Turf'] = train['Turf'].apply(lambda x: turf(x))

    train['GameWeather'] = train['GameWeather'].apply(lambda x: game_weather(x))
    train['GameWeather'] = train['GameWeather'].fillna(176418)


    train['WindDirection'] = train['WindDirection'].apply(lambda x: wind_direction(x))
    train['WindDirection'] = train['WindDirection'].fillna(325534)

    train['Temperature'] = train['Temperature'].fillna(19)
    train['Humidity'] = train['Humidity'].fillna(14)
    train['DefendersInTheBox'] = train['DefendersInTheBox'].fillna(8)

    train['WindSpeed'] = train['WindSpeed'].apply(lambda x: wind_speed(x))
    train['WindSpeed'] = train['WindSpeed'].fillna(338426)

    train['FieldPosition'] = train['FieldPosition'].apply(lambda x: team_abbr(x))
    train['FieldPosition'] = train['FieldPosition'].fillna(28710)

    train['PossessionTeam'] = train['PossessionTeam'].apply(lambda x: team_abbr(x))
    train['OffensePersonnel'] = train['OffensePersonnel'].apply(lambda x: offense_personnel(x))
    train['DefensePersonnel'] = train['DefensePersonnel'].apply(lambda x: defense_personnel(x))

    ## sort
    # train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)
    train = train.sort_values(by=['PlayId'])

    dropped_col = ['NflId', 'DisplayName', 'Season', 'GameClock', 'TimeHandoff', 'TimeSnap',
                   'PlayerBirthDate']
    train.drop(dropped_col, axis=1, inplace=True)
    train = train.fillna(-999)

    return train
