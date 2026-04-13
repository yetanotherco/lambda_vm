#import "/book.typ": project, meta

#show: project.with(title: "", cond: () => true)

#align(center, title(meta.title))
#align(center)[_Version #meta.version _]
#align(center, meta.authors.join(", "))


This is the specification for the #link("https://github.com/yetanotherco/lambda_vm/")[Lambda verifiable vm].

