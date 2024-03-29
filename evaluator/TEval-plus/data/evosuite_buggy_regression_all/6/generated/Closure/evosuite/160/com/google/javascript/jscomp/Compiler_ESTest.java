/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:07:38 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.CodeChangeHandler;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerInput;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.CssRenamingMap;
import com.google.javascript.jscomp.DefaultPassConfig;
import com.google.javascript.jscomp.ErrorManager;
import com.google.javascript.jscomp.FunctionInformationMap;
import com.google.javascript.jscomp.InlineSimpleMethods;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSModuleGraph;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PassConfig;
import com.google.javascript.jscomp.ReferenceCollectingCallback;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.ReverseAbstractInterpreter;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.SourceMap;
import com.google.javascript.jscomp.TypeCheck;
import com.google.javascript.jscomp.TypeValidator;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import java.io.ByteArrayOutputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Compiler_ESTest extends Compiler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSource((JSModule) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSourceArray();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Logger logger0 = Logger.getLogger("88G|Ww");
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      // Undeclared exception!
      try { 
        compiler0.toSource(compiler_CodeBuilder0, (-1), (Node) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSourceArray((JSModule) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      compiler0.disableThreads();
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      compiler0.compile((List<JSSourceFile>) stack0, (List<JSSourceFile>) stack0, compilerOptions0);
      assertFalse(compiler0.hasErrors());
      assertFalse(compilerOptions0.checkTypes);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule[] jSModuleArray0 = new JSModule[1];
      // Undeclared exception!
      try { 
        compiler0.init((JSSourceFile[]) null, jSModuleArray0, (CompilerOptions) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.resetUniqueNameId();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Level level0 = Level.FINER;
      Compiler.setLoggingLevel(level0);
      assertEquals(400, level0.intValue());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getCssRenamingMap();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      boolean boolean0 = compiler0.hasRegExpGlobalReferences();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("`D\"R5", "`D\"R5");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // duplicate key: modifies
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.IntermediateState compiler_IntermediateState0 = compiler0.getState();
      compiler0.setState(compiler_IntermediateState0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      // Undeclared exception!
      try { 
        compiler0.updateGlobalVarReferences((Map<Scope.Var, ReferenceCollectingCallback.ReferenceCollection>) null, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Collections$UnmodifiableCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.rebuildInputsFromModules();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.computeCFG();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VariableMap variableMap0 = compiler0.getPropertyMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ScopeCreator scopeCreator0 = compiler0.getTypedScopeCreator();
      assertNull(scopeCreator0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.compile((JSSourceFile) null, (JSSourceFile) null, (CompilerOptions) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        Normalize.parseAndNormalizeSyntheticCode(compiler0, "QB-<LA0H<1b_Y", "QB-<LA0H<1b_Y");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.processDefines();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.DefaultPassConfig", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Callable<TypeCheck> callable0 = (Callable<TypeCheck>) mock(Callable.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(callable0).call();
      TypeCheck typeCheck0 = Compiler.runCallableWithLargeStack(callable0);
      assertNull(typeCheck0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.languageMode();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.setCssRenamingMap((CssRenamingMap) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getWarningCount();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VariableMap variableMap0 = compiler0.getVariableMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FunctionInformationMap functionInformationMap0 = compiler0.getFunctionalInformationMap();
      assertNull(functionInformationMap0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      assertNotNull(supplier0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.recordFunctionInformation();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      String string0 = compiler0.toSource(node0);
      assertFalse(compiler0.isTypeCheckingEnabled());
      assertEquals("", string0);
      assertEquals(4096, node0.getSourcePosition());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModuleGraph jSModuleGraph0 = compiler0.getModuleGraph();
      assertNull(jSModuleGraph0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceMap sourceMap0 = compiler0.getSourceMap();
      assertNull(sourceMap0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.isTypeCheckingEnabled();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.getRoot();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      List<CompilerInput> list0 = compiler0.getInputsForTesting();
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("Strip code", "Strip code");
      JSModule[] jSModuleArray0 = new JSModule[0];
      // Undeclared exception!
      try { 
        compiler0.compile(jSSourceFile0, jSModuleArray0, (CompilerOptions) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager((Logger) null);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      CodeChangeHandler.RecentChange codeChangeHandler_RecentChange0 = new CodeChangeHandler.RecentChange();
      compiler0.removeChangeHandler(codeChangeHandler_RecentChange0);
      assertEquals(0, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setHasRegExpGlobalReferences(false);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.getMessages();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      List<CompilerInput> list0 = compiler0.getExternsForTesting();
      assertNull(list0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.removeTryCatchFinally();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Scope scope0 = compiler0.getTopScope();
      assertNull(scope0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ReferenceCollectingCallback.ReferenceMap referenceCollectingCallback_ReferenceMap0 = compiler0.getGlobalVarReferences();
      assertNull(referenceCollectingCallback_ReferenceMap0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLength();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      compiler0.disableThreads();
      compiler0.toSource();
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLineIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getColumnIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.reset();
      assertEquals("", compiler_CodeBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("o},@jjVW`.Nv<Q");
      Compiler compiler0 = new Compiler(mockPrintStream0);
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("", "");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // a source must have a name
         //
         verifyException("com.google.javascript.jscomp.SourceFile", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      LinkedList<JSSourceFile> linkedList0 = new LinkedList<JSSourceFile>();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("WKjBZ:M");
      linkedList0.add(0, jSSourceFile0);
      CompilerOptions compilerOptions0 = compiler0.options;
      compiler0.compile((List<JSSourceFile>) linkedList0, (List<JSSourceFile>) linkedList0, compilerOptions0);
      assertEquals(1, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      // Undeclared exception!
      try { 
        Compiler.runCallable((Callable<Object>) null, false, true);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NullPointerException
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0);
      Callable<Object> callable0 = (Callable<Object>) mock(Callable.class, new ViolatedAssumptionAnswer());
      doReturn(mockPrintStream0).when(callable0).call();
      Object object0 = Compiler.runCallable(callable0, false, true);
      assertSame(object0, mockPrintStream0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      // Undeclared exception!
      try { 
        Compiler.runCallable((Callable<Object>) null, false, false);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NullPointerException
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      DefaultPassConfig defaultPassConfig0 = new DefaultPassConfig((CompilerOptions) null);
      compiler0.setPassConfig(defaultPassConfig0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PassConfig passConfig0 = compiler0.getPassConfig();
      // Undeclared exception!
      try { 
        compiler0.setPassConfig(passConfig0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // this.passes has already been assigned
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      InlineSimpleMethods inlineSimpleMethods0 = new InlineSimpleMethods(compiler0);
      Set<String> set0 = inlineSimpleMethods0.externMethodsWithoutSignatures;
      // Undeclared exception!
      try { 
        compiler0.stripCode(set0, set0, set0, set0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[2];
      Charset charset0 = Charset.defaultCharset();
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("RB-<LA0H<1b_Y", charset0);
      jSSourceFileArray0[0] = jSSourceFile0;
      jSSourceFileArray0[1] = jSSourceFile0;
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.init(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      compiler0.startPass("// Input %num%");
      // Undeclared exception!
      try { 
        compiler0.optimize();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.endPass();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Tracer should not be null at the end of a pass.
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.reportCodeChange();
      // Undeclared exception!
      try { 
        compiler0.newTracer("lrHWY7KivN)");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("/data/lhy/TEval-plus/j3cxMr1?p6/j3cxMr1?p6");
      compiler0.areNodesEqualForInlining(node0, node0);
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode(" [testcode] ");
      // Undeclared exception!
      try { 
        compiler0.removeInput(" [testcode] ");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      CompilerOptions compilerOptions0 = new CompilerOptions();
      compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      compiler0.removeInput(" [testcode] ");
      assertFalse(compiler0.hasErrors());
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      // Undeclared exception!
      try { 
        compiler0.newExternInput("");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      LinkedList<JSModule> linkedList0 = new LinkedList<JSModule>();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      Stack<JSSourceFile> stack0 = new Stack<JSSourceFile>();
      compiler0.initModules(stack0, linkedList0, compilerOptions0);
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("Compiler", sourceFile_Generator0);
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      compiler0.addIncrementalSourceAst(jsAst0);
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      compiler0.getReverseAbstractInterpreter();
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      assertFalse(compiler0.isTypeCheckingEnabled());
      assertNotNull(reverseAbstractInterpreter0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      compiler0.getTypeValidator();
      TypeValidator typeValidator0 = compiler0.getTypeValidator();
      assertNotNull(typeValidator0);
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      Node node0 = compiler0.parseTestCode("");
      assertFalse(compiler0.isTypeCheckingEnabled());
      assertEquals(0, node0.getCharno());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.append(">C");
      boolean boolean0 = compiler_CodeBuilder0.endsWith("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("*/\n");
      assertSame(compiler_CodeBuilder0, compiler_CodeBuilder1);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      boolean boolean0 = compiler_CodeBuilder0.endsWith(".dYZy~$S{16k)l}>");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("8K,|0pKh'aolRW*C#");
      boolean boolean0 = compiler_CodeBuilder1.endsWith("RB-<LA0H<1b_Y");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.initCompilerOptionsIfTesting();
      boolean boolean0 = compiler0.isInliningForbidden();
      assertFalse(boolean0);
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode(" [testcode] ");
      boolean boolean0 = compiler0.acceptEcmaScript5();
      assertFalse(compiler0.hasErrors());
      assertFalse(compiler0.isTypeCheckingEnabled());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      String[] stringArray0 = new String[0];
      JSError jSError0 = JSError.make("?:~:<", (-2061), (-2061), compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      // Undeclared exception!
      try { 
        compiler0.report(jSError0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.throwInternalError(".dYZy~$S{16k)l}>", (Exception) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // .dYZy~$S{16k)l}>
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("7x6Ba");
      CompilerOptions compilerOptions0 = new CompilerOptions();
      // Undeclared exception!
      try { 
        compiler0.compile((List<JSSourceFile>) null, (List<JSSourceFile>) null, compilerOptions0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Region region0 = compiler0.getSourceRegion("com.google.javascript.jscomp.DefaultPassConfig$22", (-1399));
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("/data/lhy/TEval-plus/j3cxMr1?p6/j3cxMr1?p6");
      compiler0.getSourceRegion("/data/lhy/TEval-plus/j3cxMr1?p6/j3cxMr1?p6", 1481);
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("/data/lhy/TEval-plus/j3cxMr1?p6/j3cxMr1?p6");
      Region region0 = compiler0.getSourceRegion(" [testcode] ", 67108864);
      assertNull(region0);
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CompilerOptions compilerOptions0 = new CompilerOptions();
      ArrayList<JSSourceFile> arrayList0 = new ArrayList<JSSourceFile>();
      Vector<JSModule> vector0 = new Vector<JSModule>();
      compiler0.compileModules(arrayList0, vector0, compilerOptions0);
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion((JSModule) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // No inputs
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      String string0 = compiler0.getAstDotGraph();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      compiler0.getErrorManager();
      assertFalse(compiler0.isTypeCheckingEnabled());
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertFalse(compiler0.isTypeCheckingEnabled());
      assertNotNull(errorManager0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("/data/lhy/TEval-plus/j3cxMr1?p6/j3cxMr1?p6");
      TreeMap<Scope.Var, ReferenceCollectingCallback.ReferenceCollection> treeMap0 = new TreeMap<Scope.Var, ReferenceCollectingCallback.ReferenceCollection>();
      // Undeclared exception!
      try { 
        compiler0.updateGlobalVarReferences(treeMap0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.Collections$UnmodifiableCollection", e);
      }
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      TreeMap<Scope.Var, ReferenceCollectingCallback.ReferenceCollection> treeMap0 = new TreeMap<Scope.Var, ReferenceCollectingCallback.ReferenceCollection>();
      Node node0 = Node.newNumber(3483.699138070068);
      // Undeclared exception!
      try { 
        compiler0.updateGlobalVarReferences(treeMap0, node0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }
}
