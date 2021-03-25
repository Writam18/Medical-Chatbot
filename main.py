from bot import chat

print('REMDEX: I will answer your queries...If you want  to exit,type bye... :)')

W = "REMDEX: "

exit_list = ['exit','see you later','bye','quit','break']

while True:
  user_input = input()
  if user_input.lower() in exit_list:
    print('REMDEX: Bye..Take Care..Chat with you later!!')
    break
  else:
    ans = chat(user_input)
    print(W+str(ans))
