local a = 1
if true then
    a = 2
else
    a = 3
end

if false then
    a = a * 5
else
    a = a * 7
end

print(a)