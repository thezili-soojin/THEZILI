var config = {
    // Language for the mirror (currently not implemented)
    language : "ko",
    greeting : ["SmartMirror"], // An array of greetings to randomly choose from
    // forcast.io
    forcast : {
        key : "62255a23499f1952654ba0edd5981aa4", // Your forcast.io api key
        units : "auto" // See forcast.io documentation if you are getting the wrong units
    },
    // Calendar (An array of iCals)
    calendar: {
      icals : ["https://calendar.google.com/calendar/ical/thezili.soojin%40gmail.com/private-132620e6dfc6e0c2300b20a8c57af898/basic.ics"],
      maxResults: 9, // Number of calender events to display (Defaults is 9)
      maxDays: 365 // Number of days to display (Default is one year)
    },
    traffic: {
      key : "AtybLUCsarX2X09PVrcR_TMtC3jQGWVT5r2bIGpuE4Nxpo5pkkyyYtc27UEx_93k", // Bing Maps API Key
      mode : "Transit", // Possibilities: Driving / Transit / Walking
      origin : "Yangjae", // Start of your trip. Human readable address.
      destination : "Suwon", // Destination of your trip. Human readable address.
      name : "THE ZILI", // Name of your destination ex: "work"
      reload_interval : 5 // Number of minutes the information is refreshed
    },

    youtube: {
      //key:"AIzaSyDB6cX5QzUQfg-msN5g4cxz0dA9zk487Xk"
    },

    subway: {
      key:"5757454e7365756e373765636b4f42"
    },
    soundcloud: {
    	key:""
    }
}
