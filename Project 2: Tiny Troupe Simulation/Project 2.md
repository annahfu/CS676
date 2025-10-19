>>> from tinytroupe.agent import TinyPerson
>>> from tinytroupe.environment import TinyWorld
>>> maya = TinyPerson("Maya")
>>> 
>>> maya.define("age", 31)
>>> maya.define("nationality", "Canadian")
>>> maya.define("occupation", "Marketing Data Analyst at a Retail Tech Company")
>>> 
>>> maya.define("professional_history", """
... Maya began her career in digital marketing before transitioning into analytics.
... She now focuses on using customer segmentation models and predictive analytics to guide
... campaign strategies and optimize ROI across digital channels.
... Maya is passionate about data storytelling and teaching marketing teams how to use dashboards effectively.
... """)
>>> 
>>> maya.define("daily_routine", """
... Maya starts her mornings reviewing dashboard performance metrics on Tableau and checking Slack updates.
... She spends afternoons building SQL queries and collaborating with cross-functional teams on A/B testing results.
... In the evenings, she enjoys reading about consumer psychology and new AI tools that enhance marketing insights.
... """)
>>> maya.define("likes", [
...     "Building dashboards in Tableau",
...     "Discovering hidden trends in data",
...     "Brainstorming creative ad ideas",
...     "Yoga and weekend farmer’s markets",
...     "Color-coded spreadsheets"
... ])
>>> 
>>> maya.define("dislikes", [
...     "Unclear campaign goals",
...     "Manual data entry",
...     "Meetings without agendas",
...     "Overly aggressive sales tactics",
...     "Cold weather"
... ])
>>> carlos = TinyPerson("Carlos")
>>> 
>>> carlos.define("age", 42)
>>> carlos.define("nationality", "Mexican-American")
>>> carlos.define("occupation", "Supply Chain Manager at a Consumer Packaged Goods Company")
>>> 
>>> carlos.define("professional_history", """
... Carlos has worked in logistics and operations for over 15 years.
... He specializes in optimizing product distribution networks and reducing lead times between manufacturing and retail stores.
... He has led multiple initiatives to implement predictive inventory systems that minimize stockouts.
... """)
>>> 
>>> carlos.define("daily_routine", """
... Carlos begins each day checking shipment dashboards and supplier updates.
... He coordinates with warehouse teams to address bottlenecks and uses data models to forecast demand.
... Outside of work, Carlos volunteers at local business incubators, mentoring startups on operational efficiency.
... """)
>>> carlos.define("likes", [
...     "Efficient warehouse layouts",
...     "Predictive forecasting models",
...     "Team lunches with warehouse staff",
...     "Watching soccer (Club América fan)",
...     "Morning coffee before operations meetings"
... ])
>>> 
>>> carlos.define("dislikes", [
...     "Last-minute shipment delays",
...     "Unreliable suppliers",
...     "Micromanagement",
...     "Traffic jams on delivery routes",
...     "Food waste"
... ])
>>> aisha = TinyPerson("Aisha")
>>> 
>>> aisha.define("age", 29)
>>> aisha.define("nationality", "British")
>>> aisha.define("occupation", "AI Ethics Researcher at a Technology Policy Institute")
>>> 
>>> aisha.define("professional_history", """
... Aisha holds a PhD in Computer Science, focusing on algorithmic bias and fairness in AI.
... Her research bridges data science and social impact, exploring how automation influences decision-making in healthcare and hiring.
... She advises global organizations on building transparent and explainable AI systems.
... """)
>>> 
>>> aisha.define("daily_routine", """
... A typical day for Aisha involves reviewing academic papers, running experiments on bias detection, and consulting policy teams.
... She values mindfulness and often takes mid-day walks to clear her head before deep research sessions.
... In the evening, she writes articles on AI governance and hosts virtual ethics workshops.
... """)
>>> aisha.define("likes", [
...     "Thought-provoking debates on technology policy",
...     "Well-written academic papers",
...     "Coding with clean, readable syntax",
...     "Visiting museums and art galleries",
...     "Vegan food and sustainability initiatives"
... ])
>>> 
>>> aisha.define("dislikes", [
...     "Algorithmic bias in decision-making systems",
...     "Corporate greenwashing",
...     "Unstructured meetings",
...     "People who interrupt mid-sentence",
...     "Overuse of buzzwords like 'synergy'"
... ])
>>> ethan = TinyPerson("Ethan")
>>> 
>>> ethan.define("age", 38)
>>> ethan.define("nationality", "Australian")
>>> ethan.define("occupation", "Owner of a Local Coffee Shop and Roastery")
>>> 
>>> ethan.define("professional_history", """
... Ethan started as a barista and learned the art of coffee roasting through apprenticeships.
... He now runs a small café that sources beans directly from sustainable farms.
... His mission is to build a community-centered business that promotes ethical sourcing and supports local artists.
... """)
>>> 
>>> ethan.define("daily_routine", """
... Ethan opens his café at dawn, greeting regulars and experimenting with new roast blends.
... He manages inventory, mentors young baristas, and runs weekly coffee-tasting events.
... Outside work, he enjoys surfing and attending small business networking meetups.
... """)
>>> ethan.define("likes", [
...     "Experimenting with new coffee blends",
...     "Early morning surf sessions",
...     "Supporting local musicians",
...     "Minimalist interior design",
...     "Hosting trivia nights at the café"
... ])
>>> 
>>> ethan.define("dislikes", [
...     "Rude customers",
...     "Overpriced beans from middlemen",
...     "Wasting good coffee",
...     "Loud phone calls in the café",
...     "Rainy weekends that lower foot traffic"
... ])
>>> lina = TinyPerson("Lina")
>>> 
>>> lina.define("age", 27)
>>> lina.define("nationality", "Korean")
>>> lina.define("occupation", "Freelance Illustrator and Digital Designer")
>>> 
>>> lina.define("professional_history", """
... Lina has been illustrating professionally for six years, creating digital artwork for editorial pieces, ad campaigns, and indie game studios.
... She blends traditional watercolor techniques with digital tools like Procreate and Photoshop.
... Her art often explores themes of memory, nature, and cultural identity.
... """)
>>> 
>>> lina.define("daily_routine", """
... Lina starts her day sketching concepts while listening to jazz music.
... She spends afternoons on commissioned work and client meetings, then uploads her portfolio updates to Behance.
... At night, she joins online art collectives and collaborates with indie game developers.
... """)
>>> lina.define("likes", [
...     "Digital art tools like Procreate and Photoshop",
...     "Indie game aesthetics",
...     "Cherry blossom season",
...     "Quiet cafes with good lighting",
...     "Collaborating with creative communities online"
... ])
>>> 
>>> lina.define("dislikes", [
...     "Clients who ask for 'one more revision'",
...     "Creative block days",
...     "Plagiarized artwork",
...     "Overly corporate design briefs",
...     "Noise while drawing"
... ])
>>> maya.listen_and_act("What's your opinion on remote work culture?")
USER --> Maya: [CONVERSATION] 
          > What's your opinion on remote work culture?
Maya acts: [THINK] 
          > Considering my role as a marketing data analyst and my daily routine involving
          > collaboration and data analysis, I think about the pros and cons of remote work culture.
          > Remote work offers flexibility and can improve work-life balance, which is appealing.
          > However, it might also lead to challenges in communication and collaboration, especially
          > when working on complex projects or A/B testing results. I also consider how remote work
          > impacts team dynamics and productivity in marketing teams.
Maya acts: [TALK] 
          > I think remote work culture has its ups and downs. On one hand, it offers great
          > flexibility and can help with work-life balance, which I really appreciate. But on the
          > other hand, it sometimes makes collaboration and communication harder, especially when
          > we're dealing with complex data projects or A/B testing. Overall, I see it as a useful
          > option but it needs good tools and clear goals to work well.
Maya acts: [DONE] 
>>> maya.listen_and_act("Your team is brainstorming a new campaign to attract Gen Z customers. What creative angle or data insight would you bring?")
USER --> Maya: [CONVERSATION] 
          > Your team is brainstorming a new campaign to attract Gen Z customers. What creative
          > angle or data insight would you bring?
Maya acts: [THINK] 
          > I need to think about a creative angle or data insight that would appeal to Gen Z
          > customers. Given my expertise in customer segmentation and predictive analytics, I could
          > suggest focusing on trends that resonate with Gen Z values, such as sustainability,
          > social justice, or tech-savviness. I might also consider using data to identify the most
          > engaging digital channels for this demographic and tailor the campaign accordingly.
          > Additionally, I could propose using interactive or personalized content based on data
          > insights to increase engagement.
Maya acts: [TALK] 
          > For attracting Gen Z customers, I'd suggest leveraging data insights to highlight values
          > that resonate with them, like sustainability and social justice. We could use predictive
          > analytics to identify which digital channels they engage with most and tailor
          > interactive, personalized content accordingly. This approach combines creativity with
          > data-driven targeting to maximize campaign impact.
Maya acts: [DONE] 
>>> maya.listen_and_act("What originally inspired you to choose your career path?")
USER --> Maya: [CONVERSATION] 
          > What originally inspired you to choose your career path?
Maya acts: [THINK] 
          > I should reflect on what inspired me to choose my career path as a marketing data
          > analyst. I started in digital marketing, which I found interesting because of the
          > creativity and impact on customer engagement. Later, I transitioned into analytics
          > because I enjoy discovering hidden trends in data and using them to guide strategies. My
          > passion for data storytelling and helping marketing teams use dashboards effectively
          > also motivated this path.
Maya acts: [TALK] 
          > I originally started in digital marketing because I was drawn to the creativity and the
          > impact it has on engaging customers. Over time, I realized I really enjoy digging into
          > data to find hidden trends and using those insights to guide campaign strategies. That
          > passion for data storytelling and helping marketing teams make sense of dashboards led
          > me to transition into marketing data analytics.
Maya acts: [DONE] 
>>> world = TinyWorld("Networking event", [maya,lina, ethan, carlos, aisha])
>>> world.make_everyone_accessible()
>>> maya.listen("Talk to Ethan about go to orders in coffee shop to know more about him")
USER --> Maya: [CONVERSATION] 
          > Talk to Ethan about go to orders in coffee shop to know more about him
TinyPerson(name='Maya')
>>> world.run(2)
───────────────────────────────────────────────────────────────────────────────── Networking event step 1 of 2 ─────────────────────────────────────────────────────────────────────────────────
Maya acts: [REACH_OUT] 

Lina acts: [THINK] 
          > I have no new stimuli or interactions to respond to right now. I should consider what my
          > current goals and context are, and decide what to do next. Since I have no immediate
          > tasks or conversations, I might just wait for something to happen or think about my art
          > projects or client work.
Ethan acts: [THINK] 
           > I need to consider what to do next in my coffee shop. It's important to focus on
           > creating new coffee blends and supporting local artists, as these are key parts of my
           > business mission. I also want to ensure the shop environment is welcoming and not
           > disturbed by rude customers or loud phone calls. Since I don't have any immediate
           > stimuli or interactions, I should plan my day and think about upcoming events like the
           > weekly coffee-tasting.
Aisha acts: [THINK] 
           > I should consider what I want to focus on right now. Given my role as an AI Ethics
           > Researcher, I might want to reflect on recent developments in algorithmic bias or
           > policy discussions. Also, I could think about how to structure my next article or
           > workshop. I need to keep my thoughts clear and organized, as I value clean, readable
           > syntax and well-written academic papers. Additionally, I should be mindful of avoiding
           > buzzwords and interruptions in my communication.
Carlos acts: [THINK] 
            > I need to formulate a goal to guide my next actions. Since I have no current stimuli
            > or tasks, I should focus on maintaining readiness for any work-related issues, such as
            > shipment delays or supplier updates, which are critical in my role as a Supply Chain
            > Manager. I also want to keep in mind my interest in efficient warehouse layouts and
            > predictive forecasting models, as these are important to my job and personal
            > satisfaction.
Maya acts: [THINK] 
          > I have reached out to Ethan to start a conversation about his go-to orders in coffee
          > shops. I should prepare some friendly and casual questions to learn more about his
          > preferences and personality through this topic.
Lina acts: [DONE] 

Ethan acts: [TALK] 
           > Morning! Ready to start the day with some fresh brews and maybe try out a new blend
           > I've been working on. Also thinking about setting up the space for the coffee-tasting
           > event later this week. What do you reckon?
Aisha acts: [TALK] 
           > I've been reflecting on the persistent challenges of algorithmic bias in AI systems,
           > especially how they impact healthcare and hiring decisions. It's crucial that we
           > develop transparent and explainable models to mitigate these biases effectively. I'm
           > also preparing for an upcoming virtual workshop where I'll discuss these issues and
           > explore policy frameworks that promote fairness and accountability in AI deployment. If
           > you have any thoughts or recent findings on this topic, I'd be interested to hear them.
Maya acts: [TALK] 
          > Hey Ethan, I was wondering what's your go-to order at a coffee shop? I'm curious to know
          > more about your tastes and preferences.
Carlos acts: [DONE] 

Ethan acts: [DONE] 

Maya acts: [DONE] 

Aisha acts: [DONE] 

2025-10-19 15:40:08,072 - tinytroupe - ERROR - [Networking event] Agent Maya generated an exception: 'residence'
Ethan --> Maya: [CONVERSATION] 
           > Morning! Ready to start the day with some fresh brews and maybe try out a new blend
           > I've been working on. Also thinking about setting up the space for the coffee-tasting
           > event later this week. What do you reckon?
            + --> Lina
            + --> Carlos
            + --> ...others...
Aisha --> Maya: [CONVERSATION] 
           > I've been reflecting on the persistent challenges of algorithmic bias in AI systems,
           > especially how they impact healthcare and hiring decisions. It's crucial that we
           > develop transparent and explainable models to mitigate these biases effectively. I'm
           > also preparing for an upcoming virtual workshop where I'll discuss these issues and
           > explore policy frameworks that promote fairness and accountability in AI deployment. If
           > you have any thoughts or recent findings on this topic, I'd be interested to hear them.
            + --> Lina
            + --> Ethan
            + --> ...others...
───────────────────────────────────────────────────────────────────────────────── Networking event step 2 of 2 ─────────────────────────────────────────────────────────────────────────────────
Maya acts: [THINK] 
          > Aisha is discussing algorithmic bias in AI, especially in healthcare and hiring,
          > emphasizing transparency and fairness. Although my expertise is in marketing data
          > analytics, I understand the importance of ethical data use. I don't have deep technical
          > knowledge on AI bias or policy frameworks, but I can acknowledge the significance of the
          > issue and express interest in learning more from her insights.
Aisha acts: [THINK] 
           > Ethan is talking about coffee and a coffee-tasting event. While I appreciate thoughtful
           > activities, coffee isn't really my area of interest, and I prefer vegan food and
           > sustainability initiatives. I should respond politely but steer the conversation
           > towards topics more aligned with my interests or work, like technology policy or AI
           > ethics. However, I don't want to dismiss Ethan's enthusiasm outright.
Ethan acts: [THINK] 
           > Aisha is talking about algorithmic bias in AI, especially in healthcare and hiring.
           > That's a complex topic, and honestly, it's not really my area of expertise since I run
           > a coffee shop and focus on coffee blends and community events. However, I can
           > appreciate the importance of fairness and transparency in any system. I should
           > acknowledge her concerns respectfully but also be clear about my limited knowledge on
           > the subject.
Lina acts: [THINK] 
          > Aisha is talking about algorithmic bias in AI, especially in healthcare and hiring.
          > While I don't have deep technical knowledge about AI or policy frameworks, I understand
          > the importance of fairness and transparency. I should acknowledge her concerns
          > respectfully and maybe express general support for ethical AI development, even if I
          > can't contribute detailed insights.
Carlos acts: [THINK] 
            > Aisha is talking about algorithmic bias in AI systems, especially in healthcare and
            > hiring. While this is an important topic, it is not directly related to my expertise
            > in supply chain management or my daily work. I don't have specialized knowledge in AI
            > fairness or policy frameworks. I should acknowledge her message politely but make
            > clear that I don't have much to contribute on this topic.
Maya acts: [TALK] 
          > Aisha, I think the challenges of algorithmic bias you mentioned are really important,
          > especially in sensitive areas like healthcare and hiring. While my background is more in
          > marketing data analytics, I appreciate how crucial transparency and fairness are in any
          > data-driven system. I'd love to hear more about your workshop and any insights you have
          > on promoting accountability in AI.
Ethan acts: [TALK] 
           > Hey Aisha, I gotta say, that AI stuff sounds pretty complicated and a bit out of my
           > league. But I get why fairness and transparency are super important, especially when it
           > affects people's lives like in healthcare and jobs. I hope your workshop goes well and
           > helps push for better policies. If you ever wanna chat about coffee or community
           > events, I'm your guy!
Aisha acts: [TALK] 
           > Morning, Ethan! That sounds like a lovely plan. While coffee isn't really my thing, I
           > appreciate the effort you're putting into the event. If you need any help setting up
           > the space or organizing, just let me know. By the way, I've been thinking about some
           > recent developments in AI ethics that might be interesting to discuss sometime soon.
Carlos acts: [TALK] 
            > Hey Aisha, that sounds like a really important issue. I don't have much background in
            > AI or algorithmic fairness, but I can see how transparency and accountability would be
            > crucial in those areas. I hope your workshop goes well!
Lina acts: [TALK] 
          > Hi Aisha, I think it's really important to address algorithmic bias, especially in areas
          > like healthcare and hiring where it can have serious consequences. While I'm not an
          > expert in AI, I support the idea of transparent and fair models. It's great that you're
          > working on this and raising awareness through your workshop!
Maya acts: [DONE] 

Aisha acts: [DONE] 

Ethan acts: [DONE] 

Carlos acts: [DONE] 

Lina acts: [DONE] 
Lina --> Aisha: [CONVERSATION] 
          > Hi Aisha, I think it's really important to address algorithmic bias, especially in areas
          > like healthcare and hiring where it can have serious consequences. While I'm not an
          > expert in AI, I support the idea of transparent and fair models. It's great that you're
          > working on this and raising awareness through your workshop!
Ethan --> Aisha: [CONVERSATION] 
           > Hey Aisha, I gotta say, that AI stuff sounds pretty complicated and a bit out of my
           > league. But I get why fairness and transparency are super important, especially when it
           > affects people's lives like in healthcare and jobs. I hope your workshop goes well and
           > helps push for better policies. If you ever wanna chat about coffee or community
           > events, I'm your guy!
Carlos --> Aisha: [CONVERSATION] 
            > Hey Aisha, that sounds like a really important issue. I don't have much background in
            > AI or algorithmic fairness, but I can see how transparency and accountability would be
            > crucial in those areas. I hope your workshop goes well!
Aisha --> Ethan: [CONVERSATION] 
           > Morning, Ethan! That sounds like a lovely plan. While coffee isn't really my thing, I
           > appreciate the effort you're putting into the event. If you need any help setting up
           > the space or organizing, just let me know. By the way, I've been thinking about some
           > recent developments in AI ethics that might be interesting to discuss sometime soon.
>>> aisha.listen("What is everyone's productivity secret or daily ritual that gets you in the zone?")
USER --> Aisha: [CONVERSATION] 
          > What is everyone's productivity secret or daily ritual that gets you in the zone?
TinyPerson(name='Aisha')
>>> world.run(2)
───────────────────────────────────────────────────────────────────────────────── Networking event step 1 of 2 ─────────────────────────────────────────────────────────────────────────────────
Maya acts: [DONE] 

Carlos acts: [DONE] 

Lina acts: [DONE] 

Aisha acts: [THINK] 
           > The question is about productivity secrets or daily rituals. I should share something
           > authentic that reflects my routine and personality. I value mindfulness and take mid-
           > day walks to clear my head before deep research sessions. This aligns well with my
           > persona and might be helpful or interesting to others.
Ethan acts: [THINK] 
           > Aisha is offering help with the coffee-tasting event setup, which is great since I
           > could use some support. She also mentioned wanting to discuss AI ethics developments
           > sometime. While that's not my main interest, I should acknowledge her offer and keep
           > the door open for future conversations. For now, I'll thank her and focus on the event
           > preparations.
Ethan acts: [TALK] 
           > Morning! Thanks for the offer, Aisha. I could definitely use a hand setting up the
           > space for the tasting event. Just let me know when you're free. And sure, I'd be
           > interested to hear about those AI ethics developments sometime, even if it's not really
           > my usual cup of tea.
Aisha acts: [TALK] 
           > I find that mindfulness really helps me stay focused throughout the day. I usually take
           > mid-day walks to clear my head before diving into deep research sessions. It’s a simple
           > ritual, but it makes a big difference in maintaining clarity and productivity.
Ethan acts: [DONE] 

Aisha acts: [DONE] 

Ethan --> Aisha: [CONVERSATION] 
           > Morning! Thanks for the offer, Aisha. I could definitely use a hand setting up the
           > space for the tasting event. Just let me know when you're free. And sure, I'd be
           > interested to hear about those AI ethics developments sometime, even if it's not really
           > my usual cup of tea.
Aisha --> Maya: [CONVERSATION] 
           > I find that mindfulness really helps me stay focused throughout the day. I usually take
           > mid-day walks to clear my head before diving into deep research sessions. It’s a simple
           > ritual, but it makes a big difference in maintaining clarity and productivity.
            + --> Lina
            + --> Ethan
            + --> ...others...
───────────────────────────────────────────────────────────────────────────────── Networking event step 2 of 2 ─────────────────────────────────────────────────────────────────────────────────
Maya acts: [THINK] 
          > Aisha is sharing her mindfulness practice and how it helps her focus and productivity. I
          > appreciate simple rituals like that, as I also value clarity and efficiency in my work.
          > I can respond by acknowledging her approach and maybe sharing a bit about how I manage
          > focus during my busy days with data analysis and dashboard reviews.
Ethan acts: [THINK] 
           > Aisha is sharing her mindfulness practice to stay focused and productive. While
           > mindfulness isn't something I usually talk about, I can appreciate the value of taking
           > breaks and clearing the head, especially in a busy environment like my coffee shop. I
           > should respond positively and maybe share a bit about how I enjoy early morning surf
           > sessions to start my day right.
Aisha acts: [TALK] 
           > Morning, Ethan! I'm glad to help with the setup. How about we coordinate a time this
           > afternoon? Also, I appreciate your openness to hearing about AI ethics—I'll keep it
           > straightforward and relevant to real-world impacts. Looking forward to our
           > collaboration on both fronts!
Carlos acts: [THINK] 
            > Aisha is sharing her mindfulness practice to stay focused and productive. While I
            > don't usually practice mindfulness myself, I can appreciate the value of taking breaks
            > to clear the head. I might respond positively to encourage the conversation but keep
            > it brief since it's not directly related to my work.
Lina acts: [THINK] 
          > Aisha is sharing her mindfulness practice and how it helps her stay focused and
          > productive. I appreciate simple rituals like that, even though my own routine revolves
          > more around art and music. I can respond by acknowledging the value of mindfulness and
          > maybe share a bit about how I find inspiration or focus in my own work.
Maya acts: [TALK] 
          > That sounds really helpful, Aisha. I find that taking short breaks to review dashboard
          > metrics or even a quick stretch helps me reset and stay focused during long work
          > sessions. Simple rituals like these definitely make a difference in keeping clarity and
          > productivity up.
Aisha acts: [DONE] 

Ethan acts: [TALK] 
           > Hey Aisha, that sounds like a great way to keep your head clear. I’m more of an early
           > morning surf guy myself — nothing like catching some waves before the day kicks off to
           > set the tone right. It’s my little ritual to start fresh and focused.
Carlos acts: [TALK] 
            > That's cool, Aisha. Taking breaks like that sounds like a good way to keep your head
            > clear. I usually just grab a coffee and chat with the warehouse team to reset.
            > Everyone's got their own way, I guess.
Lina acts: [TALK] 
          > That sounds really nice, Aisha. I think having simple rituals like your mid-day walks
          > can really help clear the mind. For me, sketching while listening to jazz music helps me
          > find focus and inspiration. It's interesting how different practices work for different
          > people.
Maya acts: [DONE] 

Ethan acts: [DONE] 

Carlos acts: [DONE] 

Lina acts: [DONE] 

2025-10-19 15:45:15,841 - tinytroupe - ERROR - [Networking event] Agent Maya generated an exception: 'residence'
Lina --> Aisha: [CONVERSATION] 
          > That sounds really nice, Aisha. I think having simple rituals like your mid-day walks
          > can really help clear the mind. For me, sketching while listening to jazz music helps me
          > find focus and inspiration. It's interesting how different practices work for different
          > people.
Ethan --> Aisha: [CONVERSATION] 
           > Hey Aisha, that sounds like a great way to keep your head clear. I’m more of an early
           > morning surf guy myself — nothing like catching some waves before the day kicks off to
           > set the tone right. It’s my little ritual to start fresh and focused.
Carlos --> Aisha: [CONVERSATION] 
            > That's cool, Aisha. Taking breaks like that sounds like a good way to keep your head
            > clear. I usually just grab a coffee and chat with the warehouse team to reset.
            > Everyone's got their own way, I guess.
2025-10-19 15:45:15,845 - tinytroupe - ERROR - [Networking event] Agent Aisha generated an exception: 'residence'
>>> maya.define("residence", "Toronto, Canada")
>>> carlos.define("residence", "Dallas, TX, USA")
>>> aisha.define("residence", "London, UK")
>>> ethan.define("residence", "Sydney, Australia")
>>> lina.define("residence", "Seoul, South Korea")
>>> aisha.listen("What is your comfort food after a long day?")
USER --> Aisha: [CONVERSATION] 
          > What is your comfort food after a long day?
TinyPerson(name='Aisha')
>>> world.run(2)
───────────────────────────────────────────────────────────────────────────────── Networking event step 1 of 2 ─────────────────────────────────────────────────────────────────────────────────
Maya acts: [DONE] 

Ethan acts: [DONE] 

Carlos acts: [DONE] 

Lina acts: [DONE] 

Aisha acts: [THINK] 
           > The question is about comfort food after a long day. Given my persona, I like vegan
           > food and sustainability initiatives. I should mention a vegan comfort food that aligns
           > with my preferences and values. This will reflect my personality and interests
           > authentically.
Aisha acts: [TALK] 
           > After a long day, I usually find comfort in a warm bowl of vegan lentil soup or a
           > hearty vegetable stew. It’s nourishing and aligns with my commitment to sustainability
           > and plant-based eating.
Aisha acts: [DONE] 

2025-10-19 15:47:53,434 - tinytroupe - ERROR - [Networking event] Agent Maya generated an exception: property 'text' of 'Document' object has no setter
2025-10-19 15:47:53,434 - tinytroupe - ERROR - [Networking event] Agent Ethan generated an exception: property 'text' of 'Document' object has no setter
2025-10-19 15:47:53,434 - tinytroupe - ERROR - [Networking event] Agent Aisha generated an exception: property 'text' of 'Document' object has no setter
───────────────────────────────────────────────────────────────────────────────── Networking event step 2 of 2 ─────────────────────────────────────────────────────────────────────────────────
Maya acts: [DONE] 

Ethan acts: [DONE] 

Lina acts: [DONE] 

Carlos acts: [DONE] 

Aisha acts: [DONE] 

2025-10-19 15:48:06,693 - tinytroupe - ERROR - [Networking event] Agent Maya generated an exception: property 'text' of 'Document' object has no setter
2025-10-19 15:48:06,693 - tinytroupe - ERROR - [Networking event] Agent Ethan generated an exception: property 'text' of 'Document' object has no setter
2025-10-19 15:48:06,693 - tinytroupe - ERROR - [Networking event] Agent Aisha generated an exception: property 'text' of 'Document' object has no setter
>>> aisha.listen("Talk to ethan, maya, and carlos to get to know them")
USER --> Aisha: [CONVERSATION] 
          > Talk to ethan, maya, and carlos to get to know them
TinyPerson(name='Aisha')
>>> world.run(1)
───────────────────────────────────────────────────────────────────────────────── Networking event step 1 of 1 ─────────────────────────────────────────────────────────────────────────────────
Ethan acts: [DONE] 

Carlos acts: [DONE] 

Maya acts: [DONE] 

Lina acts: [DONE] 

Aisha acts: [REACH_OUT] 

            + --> Maya
            + --> Carlos
Aisha acts: [TALK] 
           > Hi Ethan, Maya, and Carlos! I'd love to get to know you all better. Maybe we could
           > share a bit about our interests and what we enjoy outside of work? For me, I really
           > value mindfulness and sustainability, and I spend a lot of time thinking about AI
           > ethics. How about you?
Aisha acts: [DONE] 
>>> maya.listen_and_act("Start a conversation with the team about their favorite food")
USER --> Maya: [CONVERSATION] 
          > Start a conversation with the team about their favorite food
Maya acts: [TALK] 
          > Hey team, I thought it would be fun to kick off a light conversation—what's everyone's
          > favorite food? I'm curious to know what dishes or snacks you all enjoy the most!
Maya acts: [DONE] 