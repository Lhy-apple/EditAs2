/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:43:47 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.common.base.Supplier;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.CssRenamingMap;
import com.google.javascript.jscomp.DiagnosticGroup;
import com.google.javascript.jscomp.ErrorManager;
import com.google.javascript.jscomp.FunctionInformationMap;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.JsAst;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.OptimizeParameters;
import com.google.javascript.jscomp.PassConfig;
import com.google.javascript.jscomp.Region;
import com.google.javascript.jscomp.ReverseAbstractInterpreter;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.SourceMap;
import com.google.javascript.jscomp.SymbolTable;
import com.google.javascript.jscomp.VariableMap;
import com.google.javascript.rhino.Node;
import com.google.protobuf.ByteString;
import java.io.ByteArrayOutputStream;
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
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSource();
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
      Compiler compiler0 = new Compiler();
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Node node0 = compiler0.parseTestCode("bBgrz&S|~`R^`#");
      compiler0.toSource(compiler_CodeBuilder0, 99, node0);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("");
      // Undeclared exception!
      try { 
        compiler0.toSourceArray(jSModule0);
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
      // Undeclared exception!
      try { 
        compiler0.normalize();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setNormalized();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.resetUniqueNameId();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("(K.iC");
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.parseInputs();
      compiler0.stripCode(compilerOptions0.stripNamePrefixes, compilerOptions0.stripNamePrefixes, compilerOptions0.stripNameSuffixes, compilerOptions0.stripNameSuffixes);
      assertEquals(3, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler.setLoggingLevel((Level) null);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
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
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Compiler.IntermediateState compiler_IntermediateState0 = compiler0.getState();
      compiler0.setState(compiler_IntermediateState0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
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
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      boolean boolean0 = compiler0.precheck();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      boolean boolean0 = compiler0.isNormalized();
      assertFalse(boolean0);
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
      // Undeclared exception!
      try { 
        compiler0.parseSyntheticCode("4");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
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
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.setUnnormalized();
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      OptimizeParameters optimizeParameters0 = new OptimizeParameters(compiler0);
      // Undeclared exception!
      try { 
        compiler0.process(optimizeParameters0);
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
  public void test20()  throws Throwable  {
      ByteString.Output byteString_Output0 = ByteString.newOutput();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteString_Output0);
      Compiler compiler0 = new Compiler(mockPrintStream0);
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
  public void test21()  throws Throwable  {
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
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      VariableMap variableMap0 = compiler0.getVariableMap();
      assertNull(variableMap0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      FunctionInformationMap functionInformationMap0 = compiler0.getFunctionalInformationMap();
      assertNull(functionInformationMap0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.parse();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Supplier<String> supplier0 = compiler0.getUniqueNameIdSupplier();
      assertNotNull(supplier0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
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
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.toSource((Node) null);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Cannot build without root node being specified
         //
         verifyException("com.google.javascript.jscomp.CodePrinter$Builder", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("4");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("4", "4");
      CompilerOptions compilerOptions0 = compiler0.options;
      JSSourceFile jSSourceFile1 = JSSourceFile.fromCode((String) null, "_*,K8R=kn\u0001lO#}WcY0");
      compiler0.compile(jSSourceFile1, jSSourceFile0, compilerOptions0);
      assertEquals(3, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("B");
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion(jSModule0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
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
      SymbolTable symbolTable0 = new SymbolTable(compiler0);
      compiler0.removeChangeHandler(symbolTable0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
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
  public void test35()  throws Throwable  {
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
  public void test36()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLength();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      String string0 = compiler_CodeBuilder0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getLineIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.reset();
      assertEquals("", compiler_CodeBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SymbolTable symbolTable0 = compiler0.acquireSymbolTable();
      assertNotNull(symbolTable0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      ByteArrayOutputStream byteArrayOutputStream0 = new ByteArrayOutputStream();
      MockPrintStream mockPrintStream0 = new MockPrintStream(byteArrayOutputStream0, false);
      Compiler compiler0 = new Compiler(mockPrintStream0);
      compiler0.parseTestCode("iG");
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("Two modules cannot contain the same input, but module {0} and {1} both include \"{2}\"");
      JSModule jSModule0 = new JSModule("HIVl#O");
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      JSModule[] jSModuleArray0 = new JSModule[1];
      jSModuleArray0[0] = jSModule0;
      compiler0.compile(jSSourceFile0, jSModuleArray0, compilerOptions0);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("4");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("4", "4");
      CompilerOptions compilerOptions0 = compiler0.options;
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[8];
      jSSourceFileArray0[0] = jSSourceFile0;
      jSSourceFileArray0[1] = jSSourceFile0;
      jSSourceFileArray0[2] = jSSourceFile0;
      jSSourceFileArray0[3] = jSSourceFile0;
      jSSourceFileArray0[4] = jSSourceFile0;
      jSSourceFileArray0[5] = jSSourceFile0;
      jSSourceFileArray0[6] = jSSourceFile0;
      jSSourceFileArray0[7] = jSSourceFile0;
      compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      compiler0.disableThreads();
      JSModule jSModule0 = new JSModule("");
      compiler0.toSource(jSModule0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode(" on recently change AST");
      // Undeclared exception!
      try { 
        compiler0.toSource((JSModule) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // java.lang.NullPointerException
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      PassConfig passConfig0 = compiler0.createPassConfigInternal();
      compiler0.setPassConfig(passConfig0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.getState();
      PassConfig passConfig0 = compiler0.createPassConfigInternal();
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
  public void test48()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("FIXED32");
      // Undeclared exception!
      try { 
        compiler0.check();
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
  public void test49()  throws Throwable  {
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
  public void test50()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.reportCodeChange();
      // Undeclared exception!
      try { 
        compiler0.newTracer("RqjT?=cAWQ*");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      compiler0.areNodesEqualForInlining(node0, node0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode(" on recently change< AST");
      // Undeclared exception!
      try { 
        compiler0.newExternInput((String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode(" on recently change AST");
      // Undeclared exception!
      try { 
        compiler0.newExternInput(" [testcode] ");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Conflicting externs name:  [testcode] 
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator(" [testcode] ", sourceFile_Generator0);
      JsAst jsAst0 = new JsAst(jSSourceFile0);
      // Undeclared exception!
      try { 
        compiler0.addIncrementalSourceAst(jsAst0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Duplicate input of name  [testcode] 
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("");
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      SourceFile sourceFile0 = SourceFile.fromGenerator("", sourceFile_Generator0);
      JsAst jsAst0 = new JsAst(sourceFile0);
      compiler0.addIncrementalSourceAst(jsAst0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      ReverseAbstractInterpreter reverseAbstractInterpreter0 = compiler0.getReverseAbstractInterpreter();
      assertNotNull(reverseAbstractInterpreter0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("Wo_wy:");
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.parseInputs();
      compiler0.parseInputs();
      assertEquals(3, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode(" on recently change< AST");
      Node node0 = compiler0.parseTestCode(" on recently change< AST");
      assertEquals(49, Node.DIRECT_EVAL);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("*/\n");
      assertSame(compiler_CodeBuilder1, compiler_CodeBuilder0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      int int0 = compiler_CodeBuilder0.getColumnIndex();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      boolean boolean0 = compiler_CodeBuilder0.endsWith("float");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      compiler_CodeBuilder0.append("JSC_OPTIMIZE_LOOP_ERROR");
      boolean boolean0 = compiler_CodeBuilder0.endsWith("S!");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Compiler.CodeBuilder compiler_CodeBuilder0 = new Compiler.CodeBuilder();
      Compiler.CodeBuilder compiler_CodeBuilder1 = compiler_CodeBuilder0.append("RqG0Hh~");
      boolean boolean0 = compiler_CodeBuilder1.endsWith("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      compiler0.optimize();
      assertFalse(compiler0.isTypeCheckingEnabled());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("4");
      boolean boolean0 = compiler0.isInliningForbidden();
      assertFalse(boolean0);
      assertFalse(compiler0.hasErrors());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("(K.iC");
      JSModule[] jSModuleArray0 = new JSModule[0];
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      DiagnosticGroup diagnosticGroup0 = DiagnosticGroup.forType(compiler0.MOTION_ITERATIONS_ERROR);
      compilerOptions0.setWarningLevel(diagnosticGroup0, compiler0.OPTIMIZE_LOOP_ERROR.level);
      compiler0.compile(jSSourceFile0, jSModuleArray0, compilerOptions0);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      String[] stringArray0 = new String[4];
      JSError jSError0 = JSError.make((String) null, node0, compilerOptions0.reportUnknownTypes, compiler0.OPTIMIZE_LOOP_ERROR, stringArray0);
      compiler0.report(jSError0);
      assertEquals(0, compiler0.getWarningCount());
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        compiler0.throwInternalError(">tZUD48Mj", (Exception) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // >tZUD48Mj
         //
         verifyException("com.google.javascript.jscomp.Compiler", e);
      }
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Region region0 = compiler0.getSourceRegion("51ypg|r", (-3));
      assertNull(region0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("(K.iC");
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.getSourceRegion((String) null, 1);
      assertEquals(2, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromFile("(K.iC");
      CompilerOptions compilerOptions0 = compiler0.getOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      compiler0.getNodeForCodeInsertion((JSModule) null);
      assertEquals(3, compiler0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      JSModule jSModule0 = new JSModule("B");
      JSSourceFile jSSourceFile0 = JSSourceFile.fromCode("undefinedVars", "undefinedVars");
      jSModule0.add(jSSourceFile0);
      // Undeclared exception!
      try { 
        compiler0.getNodeForCodeInsertion(jSModule0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      String string0 = compiler0.getAstDotGraph();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseSyntheticCode("(K.iC", "(K.iC");
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertEquals(1, errorManager0.getErrorCount());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ErrorManager errorManager0 = compiler0.getErrorManager();
      assertEquals(0, errorManager0.getErrorCount());
  }
}