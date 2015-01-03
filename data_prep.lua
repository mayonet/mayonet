require 'image'
require 'lfs'

TRAIN_DIR = [[/media/marat/MySSD/plankton/train]]

classes = {}
for d in lfs.dir(TRAIN_DIR) do
    if string.sub(d, 1, 1) ~= '.' then
        table.insert(classes, d)
    end
end
table.sort(classes)



------------------
-- iterate files using function imageProcessor that takes an image
-- and class name and filename. Function must return 0 to continue iteration
function iterateImages(imageProcessor)
    local stop = false
    for c, d in ipairs(classes) do
        if stop then
            break
        end
        local class_dir = TRAIN_DIR..'/'..d
        for f in lfs.dir(class_dir) do
            if string.sub(f, 1, 1) ~= '.' then
                local img = image.load(class_dir..'/'..f)
                if imageProcessor(img, d, f) ~= 0 then
                    stop = true
                    break
                end
            end
        end
    end
end

sz_w = {}
sz_h = {}
function saveHW(img, c, fn)
    table.insert(sz_w, (#img[1])[1])
    table.insert(sz_h, (#img[1])[2])
    return 0
end
iterateImages(saveHW)
mw = 0
for i, s in pairs(sz_w) do
    mw = math.max(s, mw)
end
print('max width: '..mw)

mh = 0
for i, s in pairs(sz_h) do
    mh = math.max(s, mh)
end
print('max heigh: '..mh)


function plotBig(img, c, fn)
    if (#img[1])[1] >= 410 then
        print(c..'/'..fn)
        image.display{image=img, legend=c}
    end
    return 0
end

iterateImages(plotBig)


