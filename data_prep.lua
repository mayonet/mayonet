require 'image'
require 'lfs'

TRAIN_DIR = [[/media/marat/MySSD/plankton/train]]

CROP_SIZE = 64

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


-----------------------------------------
------- collect sizes of all pictures in training sample
--sz_w = {}
--sz_h = {}
--function saveHW(img, c, fn)
--    table.insert(sz_w, (#img[1])[1])
--    table.insert(sz_h, (#img[1])[2])
--    return 0
--end
--iterateImages(saveHW)
--mw = 0
--for i, s in pairs(sz_w) do
--    mw = math.max(s, mw)
--end
--print('max width: '..mw)
--
--mh = 0
--for i, s in pairs(sz_h) do
--    mh = math.max(s, mh)
--end
--print('max heigh: '..mh)


-------------------------------
------- draw big pictures
-------------------------------
--function plotBig(img, c, fn)
--    if (#img[1])[1] >= 410 then
--        print(c..'/'..fn)
--        image.display{image=img, legend=c}
--    end
--    return 0
--end
--iterateImages(plotBig)

function d(img)
    image.display(img)
end

--function inverseImage(img)
--    return img:clone():fill(1.0):add(-img)
--end

-----------------------
-- crop image to 1xCROP_SIZExCROP_SIZE
function normalizeImage(img)
    local cimg = img:clone()
    local h = (#cimg)[2]
    local w = (#cimg)[3]

    if w > CROP_SIZE or h > CROP_SIZE then
        local xOffset = math.max(0, math.floor((w-CROP_SIZE)/2))
        local yOffset = math.max(0, math.floor((h-CROP_SIZE)/2))
        w = math.min(CROP_SIZE, w)
        h = math.min(CROP_SIZE, h)
        print((1 + xOffset)..' '..(1 + yOffset)..' '..(w + xOffset)..' '..(h + yOffset)..' w='..w..' h='..h)
        cimg = image.crop(cimg, 1 + xOffset, 1 + yOffset, w + xOffset + 1, h + yOffset + 1)
    end

    local nh = (#cimg)[2]
    local nw = (#cimg)[3]
    print('new_h='..nh..' new_w='..nw)

    local nimg = cimg:clone():resize(1, CROP_SIZE, CROP_SIZE):new():fill(1.0)
    local xOffset = math.floor((CROP_SIZE-w)/2)
    local yOffset = math.floor((CROP_SIZE-h)/2)
    print('xOffset='..xOffset..'  yOffset='..yOffset..'  h='..h..'  w='..w)
    nimg[{1, {1+yOffset, h+yOffset}, {1+xOffset,w+xOffset}}] = cimg
    return nimg
end

