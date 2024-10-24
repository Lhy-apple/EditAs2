/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:46:56 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CheckGlobalThis;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CheckMissingGetCssName;
import com.google.javascript.jscomp.CoalesceVariableNames;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConvertToDottedProperties;
import com.google.javascript.jscomp.FieldCleanupPass;
import com.google.javascript.jscomp.InlineSimpleMethods;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.LoggerErrorManager;
import com.google.javascript.jscomp.MessageFormatter;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.OptimizeArgumentsArray;
import com.google.javascript.jscomp.RenameLabels;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.StrictModeCheck;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.InputId;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Stack;
import java.util.logging.Logger;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NodeTraversal_ESTest extends NodeTraversal_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("cm.google.javascriptjscomp.NodeTravesal");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "cm.google.javascriptjscomp.NodeTravesal");
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) fieldCleanupPass_QualifiedNameSearchTraversal0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("NJ#-{vU");
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      nodeTraversal0.traverseAtScope(scope0);
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Logger logger0 = Logger.getGlobal();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager((MessageFormatter) null, logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckMissingGetCssName checkMissingGetCssName0 = new CheckMissingGetCssName(compiler0, checkLevel0, "undefinedNames");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkMissingGetCssName0, (ScopeCreator) null);
      InputId inputId0 = nodeTraversal0.getInputId();
      assertNull(inputId0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Logger logger0 = Logger.getAnonymousLogger();
      LoggerErrorManager loggerErrorManager0 = new LoggerErrorManager(logger0);
      Compiler compiler0 = new Compiler(loggerErrorManager0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      JSError jSError0 = nodeTraversal0.makeError((Node) null, compiler0.MOTION_ITERATIONS_ERROR, (String[]) null);
      assertEquals(0, jSError0.getNodeLength());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      ConvertToDottedProperties convertToDottedProperties0 = new ConvertToDottedProperties(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, convertToDottedProperties0);
      Node node0 = nodeTraversal0.getCurrentNode();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.NodeTraversal");
      StrictModeCheck strictModeCheck0 = new StrictModeCheck(compiler0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) strictModeCheck0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      Compiler compiler1 = nodeTraversal0.getCompiler();
      assertSame(compiler0, compiler1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("PH_Xt_Be");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("PH_Xt_Be");
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      String[] stringArray0 = new String[0];
      JSError jSError0 = nodeTraversal0.makeError(node0, checkLevel0, compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      assertEquals(1, jSError0.lineNumber);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = Node.newString(105, "a", 4604, 105);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "");
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, (Node) null, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = Node.newString(105, "a", 4604, 105);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "a");
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, node0, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("PH_Xt_Be");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createInitialScope(node0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("DvTNv");
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Stack<Node> stack0 = new Stack<Node>();
      NodeTraversal.traverseRoots((AbstractCompiler) compiler0, (List<Node>) stack0, (NodeTraversal.Callback) optimizeArgumentsArray0);
      assertEquals(10, stack0.capacity());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("ObjectPropertyStringPreprocess", "ObjectPropertyStringPreprocess");
      RenameLabels renameLabels0 = new RenameLabels(compiler0);
      RenameLabels.ProcessLabels renameLabels_ProcessLabels0 = renameLabels0.new ProcessLabels();
      Node[] nodeArray0 = new Node[1];
      nodeArray0[0] = node0;
      // Undeclared exception!
      try { 
        NodeTraversal.traverseRoots((AbstractCompiler) compiler0, (NodeTraversal.Callback) renameLabels_ProcessLabels0, nodeArray0);
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
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node[] nodeArray0 = new Node[2];
      Node node0 = new Node(85);
      nodeArray0[0] = node0;
      Node node1 = new Node((-43), node0, node0, 49, 53);
      nodeArray0[1] = node1;
      // Undeclared exception!
      try { 
        NodeTraversal.traverseRoots((AbstractCompiler) compiler0, (NodeTraversal.Callback) optimizeArgumentsArray0, nodeArray0);
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
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = Node.newNumber((-659.2792814165));
      ConvertToDottedProperties convertToDottedProperties0 = new ConvertToDottedProperties(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, convertToDottedProperties0);
      nodeTraversal0.traverseInnerNode(node0, node0, (Scope) null);
      assertFalse(node0.isArrayLit());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("[*Fz;!");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverseInnerNode(node0, node0, scope0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = compiler0.parseTestCode("ee,fZwzEA[p@HI>}");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = compiler0.parseTestCode("ee,fZwzEA[p@HI>}");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      Node node1 = new Node(37, node0, 50, 2);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(50, int0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = compiler0.parseTestCode("%nae%");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      nodeTraversal0.getModule();
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      compiler0.parseTestCode("");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      JSModule jSModule0 = nodeTraversal0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = new Node(105);
      Node node1 = new Node(4, node0, node0, 36, 176);
      Node[] nodeArray0 = new Node[1];
      nodeArray0[0] = node0;
      // Undeclared exception!
      try { 
        NodeTraversal.traverseRoots((AbstractCompiler) compiler0, (NodeTraversal.Callback) optimizeArgumentsArray0, nodeArray0);
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
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      InlineSimpleMethods inlineSimpleMethods0 = new InlineSimpleMethods(compiler0);
      NodeTraversal.Callback nodeTraversal_Callback0 = inlineSimpleMethods0.getActingCallback();
      Node node0 = Node.newString(105, "#", 105, 105);
      Node node1 = Node.newString(105, "#", 105, 105);
      Node node2 = Node.newString(105, "#", 105, 105);
      Node node3 = new Node(105, node1, node2, node0, node0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node3, nodeTraversal_Callback0);
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
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      Node node0 = nodeTraversal0.getEnclosingFunction();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      // Undeclared exception!
      try { 
        typedScopeCreator0.createScope(node0, (Scope) null);
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
  public void test26()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = compiler0.parseTestCode("?:<rAiPY23!r6");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
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
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.NodeTraversal$AbstractScopedCallback", "com.google.javascript.jscomp.NodeTraversal$AbstractScopedCallback");
      CoalesceVariableNames coalesceVariableNames0 = new CoalesceVariableNames(compiler0, true);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) coalesceVariableNames0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
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
  public void test29()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      boolean boolean0 = nodeTraversal0.hasScope();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0);
      Node node0 = Node.newString(105, "#", 105, 105);
      Node node1 = new Node(132, node0, node0, node0, node0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node1, (NodeTraversal.Callback) optimizeArgumentsArray0);
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
}
