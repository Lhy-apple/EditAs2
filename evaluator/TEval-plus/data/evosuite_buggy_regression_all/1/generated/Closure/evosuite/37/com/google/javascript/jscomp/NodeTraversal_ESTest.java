/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:05:49 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CheckAccidentalSemicolon;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.ClosureCodingConvention;
import com.google.javascript.jscomp.CodingConvention;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.CompilerOptions;
import com.google.javascript.jscomp.FieldCleanupPass;
import com.google.javascript.jscomp.GroupVariableDeclarations;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.JSSourceFile;
import com.google.javascript.jscomp.MinimizeExitPoints;
import com.google.javascript.jscomp.MoveFunctionDeclarations;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.PrintStreamErrorManager;
import com.google.javascript.jscomp.RenameLabels;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.SourceFile;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.jscomp.VarCheck;
import com.google.javascript.rhino.ErrorReporter;
import com.google.javascript.rhino.InputId;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NodeTraversal_ESTest extends NodeTraversal_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("1A");
      Normalize.PropagateConstantAnnotationsOverVars normalize_PropagateConstantAnnotationsOverVars0 = new Normalize.PropagateConstantAnnotationsOverVars(compiler0, true);
      // Undeclared exception!
      try { 
        normalize_PropagateConstantAnnotationsOverVars0.process(node0, node0);
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
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      Node node0 = compiler0.parseTestCode("");
      Scope scope0 = new Scope(node0, compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, groupVariableDeclarations0);
      nodeTraversal0.traverseAtScope(scope0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckAccidentalSemicolon checkAccidentalSemicolon0 = new CheckAccidentalSemicolon(checkLevel0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkAccidentalSemicolon0);
      InputId inputId0 = nodeTraversal0.getInputId();
      assertNull(inputId0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, moveFunctionDeclarations0);
      JSError jSError0 = nodeTraversal0.makeError((Node) null, compiler0.OPTIMIZE_LOOP_ERROR, (String[]) null);
      assertEquals((-1), jSError0.getCharno());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        Normalize.parseAndNormalizeTestCode(compiler0, "tURKzEe6Fbk=S:z", "tURKzEe6Fbk=S:z");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, moveFunctionDeclarations0);
      Node node0 = nodeTraversal0.getCurrentNode();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.NodeTraversal$AbstractShallowCallback");
      VarCheck varCheck0 = new VarCheck(compiler0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) varCheck0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, groupVariableDeclarations0);
      Compiler compiler1 = nodeTraversal0.getCompiler();
      assertSame(compiler0, compiler1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      RenameLabels renameLabels0 = new RenameLabels((AbstractCompiler) null);
      RenameLabels.ProcessLabels renameLabels_ProcessLabels0 = renameLabels0.new ProcessLabels();
      // Undeclared exception!
      try { 
        NodeTraversal.traverseRoots((AbstractCompiler) null, (NodeTraversal.Callback) renameLabels_ProcessLabels0, (Node[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations(compiler0);
      LinkedList<Node> linkedList0 = new LinkedList<Node>();
      NodeTraversal.traverseRoots((AbstractCompiler) compiler0, (List<Node>) linkedList0, (NodeTraversal.Callback) moveFunctionDeclarations0);
      assertEquals(0.0, compiler0.getProgress(), 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, moveFunctionDeclarations0);
      Node node0 = Node.newString("com.google.javascript.jscomp.NodeTraversal$AbstractShallowStatementCallback");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      JSError jSError0 = nodeTraversal0.makeError(node0, checkLevel0, nodeTraversal0.NODE_TRAVERSAL_ERROR, (String[]) null);
      assertEquals(0, jSError0.getNodeLength());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("Parent");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, (CodingConvention) null);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createInitialScope((Node) null);
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
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SourceFile.Generator sourceFile_Generator0 = mock(SourceFile.Generator.class, new ViolatedAssumptionAnswer());
      JSSourceFile jSSourceFile0 = JSSourceFile.fromGenerator("{0}", sourceFile_Generator0);
      CompilerOptions compilerOptions0 = compiler0.newCompilerOptions();
      compiler0.compile(jSSourceFile0, jSSourceFile0, compilerOptions0);
      ClosureCodingConvention closureCodingConvention0 = new ClosureCodingConvention();
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, closureCodingConvention0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope((Node) null, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.TypedScopeCreator", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, moveFunctionDeclarations0);
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry((ErrorReporter) null);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "PSi(R_g{");
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, (Node) null, (Node) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, minimizeExitPoints0);
      Vector<JSType> vector0 = new Vector<JSType>();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) vector0);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "3wm@7KdcR");
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, minimizeExitPoints0);
      Vector<JSType> vector0 = new Vector<JSType>();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) vector0);
      Node node1 = new Node(105, node0, node0, (-1), 1033);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "3wm@7KdcR");
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "[source unknown]");
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, minimizeExitPoints0);
      Vector<JSType> vector0 = new Vector<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) vector0);
      Node node1 = new Node(105, node0, node0, 2, 47);
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, node1, node1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      Compiler compiler1 = new Compiler();
      Node node0 = compiler1.parseSyntheticCode("P2(T53Kj='|<5cnKyno", "P2(T53Kj='|<5cnKyno");
      // Undeclared exception!
      try { 
        groupVariableDeclarations0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("1A");
      Normalize.PropagateConstantAnnotationsOverVars normalize_PropagateConstantAnnotationsOverVars0 = new Normalize.PropagateConstantAnnotationsOverVars(compiler0, true);
      Node node1 = new Node(130, node0, 38, 2);
      // Undeclared exception!
      try { 
        normalize_PropagateConstantAnnotationsOverVars0.process(node0, node1);
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
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newString("1A");
      Normalize.PropagateConstantAnnotationsOverVars normalize_PropagateConstantAnnotationsOverVars0 = new Normalize.PropagateConstantAnnotationsOverVars(compiler0, true);
      Node node1 = new Node(43, node0, node0, 51, 0);
      normalize_PropagateConstantAnnotationsOverVars0.process(node0, node0);
      assertFalse(node0.isCatch());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, groupVariableDeclarations0);
      Node node0 = compiler0.parseTestCode("uVma~h`e.[S`tW{|@}e");
      nodeTraversal0.traverseInnerNode(node0, node0, (Scope) null);
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, groupVariableDeclarations0);
      Node node0 = compiler0.parseTestCode("uVmae.[S`tW{|@}e");
      Scope scope0 = new Scope(node0, compiler0);
      nodeTraversal0.traverseAtScope(scope0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, groupVariableDeclarations0);
      Node node0 = compiler0.parseTestCode("=Z?%70/{wkI0aF**k");
      Scope scope0 = new Scope(node0, compiler0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      nodeTraversal0.getModule();
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, groupVariableDeclarations0);
      compiler0.parseTestCode("uVmae.[S`tW{|@}e");
      JSModule jSModule0 = nodeTraversal0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations(compiler0);
      Node node0 = compiler0.parseSyntheticCode("Sn$ED=gbxT", "Sn$ED=gbxT");
      Scope scope0 = new Scope(node0, compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, moveFunctionDeclarations0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertEquals("Sn$ED=gbxT", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("Parent");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, (CodingConvention) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      Node node0 = nodeTraversal0.getEnclosingFunction();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations(compiler0);
      Node node0 = compiler0.parseSyntheticCode("Sn$ED=gbxT", "Sn$ED=gbxT");
      Scope scope0 = new Scope((Node) null, compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, moveFunctionDeclarations0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverseInnerNode((Node) null, node0, scope0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      MoveFunctionDeclarations moveFunctionDeclarations0 = new MoveFunctionDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, moveFunctionDeclarations0);
      // Undeclared exception!
      try { 
        nodeTraversal0.getControlFlowGraph();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      MockPrintStream mockPrintStream0 = new MockPrintStream("Parent");
      PrintStreamErrorManager printStreamErrorManager0 = new PrintStreamErrorManager(mockPrintStream0);
      Compiler compiler0 = new Compiler(printStreamErrorManager0);
      CompilerOptions compilerOptions0 = new CompilerOptions();
      JSSourceFile[] jSSourceFileArray0 = new JSSourceFile[0];
      compiler0.compile(jSSourceFileArray0, jSSourceFileArray0, compilerOptions0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0, (CodingConvention) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null, typedScopeCreator0);
      boolean boolean0 = nodeTraversal0.hasScope();
      assertFalse(boolean0);
  }
}