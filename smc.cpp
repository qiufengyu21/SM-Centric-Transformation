#include <iostream>
#include <string>
#include <fstream>
#include <sstream>


#include "clang/AST/AST.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;

// global variable for counting how many times a CUDA kernel function has been called.
int kernelCount = 0;
int traverseCount = 1;
int gridX = 0;
int gridY = 0;
std::string gridValueX;
std::string gridValueY;
bool isDirectGridSizeInit = true;

SourceLocation sl;
int num_parents = 0;
int loop = 0;
std::string kernel_grid = "";
//CompilerInstance *CI;
SourceManager *SM;
LangOptions *LO;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");


class MyRecursiveASTVisitor : public RecursiveASTVisitor <MyRecursiveASTVisitor> {
public:
	explicit MyRecursiveASTVisitor(Rewriter &R, ASTContext *Context):Rewrite(R), Context(Context){}
	bool VisitFunctionDecl(Decl *Declaration);
	void RewriteBlockIdx(Stmt *s){
		if(MemberExpr *me = dyn_cast<MemberExpr>(s)){
			//std::string member = me->getMemberDecl()->getNameAsString();
			//me->dump();
			std::string member = me->getMemberDecl()->getNameAsString();
			if(OpaqueValueExpr *ove = dyn_cast<OpaqueValueExpr>(me->getBase())){
				Expr *SE = ove->getSourceExpr();
				if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(SE)) {
					std::string base = DRE->getNameInfo().getAsString();	
					if(base == "blockIdx" && member == "__fetch_builtin_x"){
						Rewrite.ReplaceText(me->getLocStart(), 10, "(int)fmodf((float)__SMC_chunkID, (float)__SMC_orgGridDim.x)");
					}
					if(base == "blockIdx" && member == "__fetch_builtin_y"){
						Rewrite.ReplaceText(me->getLocStart(), 10, "(int)(__SMC_chunkID/__SMC_orgGridDim.x)");
					}
					//std::cout<<base<<"."<<member<<"\n";
				}
			}
			//std::cout<<"This is a MemberExpr in kernel function! Member name: ";
			//std::cout<<me->getMemberNameInfo().getName().getAsString() << "\n";
		}
		for(Stmt::child_iterator CI = s->child_begin(), CE = s->child_end(); CI != CE; ++CI){
			if (*CI) RewriteBlockIdx(*CI);
		}
	}

	// try to find the first CUDAKernelCallExpr's parent and
	// to see if this Expr has any parent if/for/while stmt.
	// hope this works
	//const clang::Stmt* GetParentStmt(const clang::Stmt& stmt){
	int GetParentStmt(const clang::Stmt& stmt){
		auto it = Context->getParents(stmt).begin();
		if(it == Context->getParents(stmt).end()){
			return 1;
		}
		else{
			const clang::Stmt *s = it->get<clang::Stmt>();
			if(s){
				num_parents++;
				return GetParentStmt(*s);
			}
			return 1;
		}
		
		return 0;
	}

	void GetStmt(int num_parents, const clang::Stmt& stmt){
		if(loop == num_parents-2){
			auto it = Context->getParents(stmt).begin();
			const clang::Stmt *s = it->get<clang::Stmt>();
			//if(const clang::ForStmt *forstmt = dyn_cast<ForStmt>(s)){
				//std::cout<<"it's a for stmt!\n";
				
			//}
			Rewrite.InsertText(s->getLocStart(), "__SMC_init();\n", true, true);
		}
		else{
			auto it = Context->getParents(stmt).begin();
			const clang::Stmt *s = it->get<clang::Stmt>();
			loop++;
			return GetStmt(num_parents, *s);
		}
		
	}

	void RewriteKernelCall(Stmt *s){
		if(traverseCount == 2){// second time traversing the AST tree
			if(gridX == 1 && gridY == 1){
				SourceLocation sl = s->getLocStart();
				std::stringstream gridVariable;
				gridVariable << "dim3 "
					     << "__SMC_orgGridDim"
					     << " ("
					     << gridValueX
					     << ", "
					     << gridValueY
					     << ");\n";
				Rewrite.InsertText(sl, gridVariable.str(), true, true);
				traverseCount++;
				isDirectGridSizeInit = false;
				return;

			}
			else if(BinaryOperator *bo = dyn_cast<BinaryOperator>(s)){
				std::string LHS = getStmtText(bo->getLHS());
				std::string RHS = getStmtText(bo->getRHS());
				std::stringstream gridNameX;
				gridNameX<< kernel_grid << ".x";
				std::stringstream gridNameY;
				gridNameY<< kernel_grid << ".y";

				if(LHS == gridNameX.str()){
					gridX++;
					gridValueX = RHS;
				}
				else if (LHS == gridNameY.str()){
					gridY++;
					gridValueY = RHS;
				}
			}
		}
		else if(CUDAKernelCallExpr *kce = dyn_cast<CUDAKernelCallExpr>(s)){
			if(traverseCount != 1){
				return;
			}
			kernelCount++;
			//std::cout<<"KernelCount = "<<kernelCount<<"\n";
			CallExpr *kernelConfig = kce->getConfig();
			Expr *grid = kernelConfig->getArg(0);
			kernel_grid = getStmtText(grid);
			//std::cout<<kernel_grid<<"\n";
			Rewrite.ReplaceText(grid->getLocStart(), kernel_grid.length(), "grid");
			//std::cout<<"Finished rewrite grid in <<<>>>\n";
			if(kernelCount == 1){
				//std::cout<<"Inside kerNelCount == 1 \n";
				//Rewrite.InsertText(kce->getLocStart(), "__SMC_init();\n", true, true);
				int result = GetParentStmt(*s);
				//std::cout<<"Now check if result == 1 \n";
				if(result == 1){
					//std::cout<<num_parents<<"\n";
					if(num_parents <= 1){
						//std::cout<<"num_parents <= 1 \n";
						Rewrite.InsertText(kce->getLocStart(), "__SMC_init();\n", true, true);
					}
					else{
						//std::cout<<"else\n";
						GetStmt(num_parents, *s);
					}
				}
			}
			//std::cout<<"Finished adding __SMC_init();\n=====================\n";
			//Rewrite.InsertText(kce->getLocStart(), "__SMC_init();\n", true, true);
			int num_args = kce->getNumArgs();
			if(num_args == 0){
				Rewrite.InsertText(kce->getRParenLoc(), "__SMC_orgGridDim, __SMC_workersNeeded, __SMC_workerCount, __SMC_newChunkSeq, __SMC_seqEnds", true, true);
			}
			else{
				Rewrite.InsertText(kce->getRParenLoc(), ", __SMC_orgGridDim, __SMC_workersNeeded, __SMC_workerCount, __SMC_newChunkSeq, __SMC_seqEnds", true, true);
			}
			//Rewrite.InsertText(kce->getRParenLoc(), ", __SMC_orgGridDim, __SMC_workersNeeded, __SMC_workerCount,__SMC_newChunkSeq, __SMC_seqEnds", true, true);
		}

		for(Stmt::child_iterator CI = s->child_begin(), CE = s->child_end(); CI != CE; ++CI){
			if (*CI) RewriteKernelCall(*CI);
		
		}
	}

	std::string getStmtText(Stmt *s) {
		SourceLocation a(SM->getExpansionLoc(s->getLocStart())), b(Lexer::getLocForEndOfToken(SourceLocation(SM->getExpansionLoc(s->getLocEnd())), 0,  *SM, *LO));
		return std::string(SM->getCharacterData(a), SM->getCharacterData(b)-SM->getCharacterData(a));
	    }

private:
	Rewriter &Rewrite;
	ASTContext *Context;
};


bool MyRecursiveASTVisitor::VisitFunctionDecl(Decl *Declaration){
	if(traverseCount == 2){ // second time traversing the AST tree
		if(FunctionDecl *f = dyn_cast<FunctionDecl>(Declaration)){
			kernelCount = 0;
			num_parents = 0;
			loop = 0;
			if(f->hasAttr<CUDAGlobalAttr>()){
				;
			}
			else{ // this FunctionDecl is not a CUDA kernel function declaration
				if(f->doesThisDeclarationHaveABody()){
					if(Stmt *s = f->getBody()){
						RewriteKernelCall(s);
					}
				}
			}
		}
		
	}
	else if(FunctionDecl *f = dyn_cast<FunctionDecl>(Declaration)){
		kernelCount = 0;
		num_parents = 0;
		loop = 0;
		if(f->hasAttr<CUDAGlobalAttr>()){
			// we found a FunctionDecl with __global__ attribute
			// which means this is a CUDA kernel function declaration
			TypeSourceInfo *tsi = f->getTypeSourceInfo();
			TypeLoc tl = tsi->getTypeLoc();
			FunctionTypeLoc FTL = tl.getAsAdjusted<FunctionTypeLoc>();
			if(f->getNumParams() == 0){
				Rewrite.InsertText(FTL.getRParenLoc(), "dim3 __SMC_orgGridDim, int __SMC_workersNeeded, int *__SMC_workerCount, int * __SMC_newChunkSeq, int * __SMC_seqEnds", true, true);
			}
			else{
				Rewrite.InsertText(FTL.getRParenLoc(), ", dim3 __SMC_orgGridDim, int __SMC_workersNeeded, int *__SMC_workerCount, int * __SMC_newChunkSeq, int * __SMC_seqEnds", true, true);
			}
			//Rewrite.InsertText(FTL.getRParenLoc(), ", dim3 __SMC_orgGridDim, int __SMC_workersNeeded, int *__SMC_workerCount, int * __SMC_newChunkSeq, int * __SMC_seqEnds", true, true);




			if(f->doesThisDeclarationHaveABody()){
				Stmt *FuncBody = f->getBody();
				Rewrite.InsertText(FuncBody->getLocStart().getLocWithOffset(1), "\n    __SMC_Begin\n", true, true);
				Rewrite.InsertText(FuncBody->getLocEnd(), "\n    __SMC_End\n", true, true);
				if(Stmt *s = f->getBody()){
					RewriteBlockIdx(s);
				}
			}
		}
		
		else{ // this FunctionDecl is not a CUDA kernel function declaration
			if(f->doesThisDeclarationHaveABody()){
				if(Stmt *s = f->getBody()){
					RewriteKernelCall(s);		
				}
			}
		}
	}
	return true;
}


class GridHandler : public MatchFinder::MatchCallback{
public:
	GridHandler(Rewriter &Rewrite) : Rewrite(Rewrite){}
	virtual void run(const MatchFinder::MatchResult &Result){
		if(const VarDecl *vd = Result.Nodes.getNodeAs<VarDecl>("gridcall")){
			SourceLocation source = vd->getInit()->getLocStart();
			if(sl != source){
				sl  = vd->getInit()->getLocStart();
				Rewrite.ReplaceText(sl, kernel_grid.length(), "__SMC_orgGridDim");
			}
			
			//SourceLocation sl = vd->getInit()->getLocStart();
			//std::cout<<"About to add __SMC_orgGridDim!\n";
			//std::cout<<kernel_grid<<"\n";
			//Rewrite.ReplaceText(sl, kernel_grid.length(), "__SMC_orgGridDim");
		}
	}

private:
	Rewriter &Rewrite;
};


class MyASTConsumer: public ASTConsumer {
public:

	explicit MyASTConsumer(Rewriter &Rewrite, ASTContext *Context, CompilerInstance *comp) : rv(Rewrite, Context), HandleGrid(Rewrite), CI(comp){
		SourceLocation startOfFile = Rewrite.getSourceMgr().getLocForStartOfFile(Rewrite.getSourceMgr().getMainFileID());
		Rewrite.InsertText(startOfFile, "/* Added smc.h*/ \n#include \"smc.h\"\n\n",true,true);
		//Matcher.addMatcher(varDecl(hasName(kernel_grid)).bind("gridcall"), &HandleGrid);
	}

	virtual void Initialize(ASTContext &Context) {
		SM = &Context.getSourceManager();
		LO = &CI->getLangOpts();

	}

	virtual void HandleTranslationUnit(ASTContext &Context) {
		rv.TraverseDecl(Context.getTranslationUnitDecl());
		traverseCount++;
		rv.TraverseDecl(Context.getTranslationUnitDecl());
		if(isDirectGridSizeInit){
			Matcher.addMatcher(varDecl(hasName(kernel_grid)).bind("gridcall"), &HandleGrid);
			Matcher.matchAST(Context);
		}
		//Matcher.addMatcher(varDecl(hasName(kernel_grid)).bind("gridcall"), &HandleGrid);
		//Matcher.matchAST(Context);
	}

private:
	MyRecursiveASTVisitor rv;
	GridHandler HandleGrid;
	MatchFinder Matcher;
	CompilerInstance *CI;
	//SourceManager *SM;
	//LangOptions *LO;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
	MyFrontendAction() {}
	void EndSourceFileAction() override {
		const RewriteBuffer *RewriteBuf = TheRewriter.getRewriteBufferFor(TheRewriter.getSourceMgr().getMainFileID());
		//TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
		//.write(llvm::outs());
		std::ofstream outputFile;
		filename = std::string(getCurrentFile());
		filename.insert(filename.length() - 3, "_smc");
		if (!filename.empty()){
			outputFile.open(filename);
		}
		else{
			outputFile.open("output.cu");
		}

		outputFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
		outputFile.close();
	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
	StringRef file) override {
		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<MyASTConsumer>(TheRewriter, &CI.getASTContext(), &CI);
	}

private:
	Rewriter TheRewriter;
	std::string filename;
};


int main(int argc, const char **argv) {
	CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
	ClangTool Tool(OptionsParser.getCompilations(),
	OptionsParser.getSourcePathList());

	return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
