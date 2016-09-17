var config = {
    // Language for the mirror (currently not implemented)
    language : "ko-KR",
    greeting : ["Talking Mirror"], // An array of greetings to randomly choose from

    geoPosition: {
       latitude: 37.56621,
       longitude: 126.9779
    },

    // forecast.io
    forecast : {
        key : "555c62055a9f463ab184ed03f83d24b0", // Your forecast.io api key
        units : "auto", // See forecast.io documentation if you are getting the wrong units
	refreshInterval : 2
    },
    // Calendar (An array of iCals)
    calendar: {
      icals : ["https://calendar.google.com/calendar/ical/thezili.eunsook%40gmail.com/public/basic.ics"],
      maxResults: 9, // Number of calender events to display (Defaults is 9)
      maxDays: 365 // Number of days to display (Default is one year)
    },
    traffic: {
      key : "AsxasBkNzjpvAR88OT-KZy9BSo6z5DEoYfxZxyCS48GtzNAX0w-kWEMvqi5SNxLi", // Bing Maps API Key
      mode : "Transit", // Possibilities: Driving / Transit / Walking
      origin : "Suwon", // Start of your trip. Human readable address.
      destination : "Yangjae", // Destination of your trip. Human readable address.
      name : "THE ZILI", // Name of your destination ex: "work"
      reload_interval : 5 // Number of minutes the information is refreshed
    },

    youtube: {
      key:"AIzaSyD27juEEqWKgSkFwLJbJReakbfFCdWST1I"
    },
    soundcloud: {
    	//key:"28aba48328a6313e7e308beab1d2990a"
	key:"vy2u1t34bo123bu41234yduv1234tb"
    }
}
