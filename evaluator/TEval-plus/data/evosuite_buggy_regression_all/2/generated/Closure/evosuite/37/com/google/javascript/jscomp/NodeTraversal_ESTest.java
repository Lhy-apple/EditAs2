/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:27:12 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.AstParallelizer;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ExpandJqueryAliases;
import com.google.javascript.jscomp.ExternExportsPass;
import com.google.javascript.jscomp.FieldCleanupPass;
import com.google.javascript.jscomp.FindExportableNodes;
import com.google.javascript.jscomp.GatherRawExports;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.MakeDeclaredNamesUnique;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.jscomp.OptimizeArgumentsArray;
import com.google.javascript.jscomp.ReferenceCollectingCallback;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.jscomp.VarCheck;
import com.google.javascript.rhino.InputId;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.FunctionType;
import com.google.javascript.rhino.jstype.JSTypeNative;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NodeTraversal_ESTest extends NodeTraversal_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.common.codlect.LinkedListMultiqap$AsMapEntries");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "com.google.common.codlect.LinkedListMultiqap$AsMapEntries");
      Scope scope0 = new Scope(node0, compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0, (ScopeCreator) null);
      nodeTraversal0.traverseAtScope(scope0);
      nodeTraversal0.getModule();
      assertEquals(1, nodeTraversal0.getLineNumber());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      GatherRawExports gatherRawExports0 = new GatherRawExports(compiler0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverseRoots((AbstractCompiler) compiler0, (NodeTraversal.Callback) gatherRawExports0, (Node[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0, "q^BHS");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0, (ScopeCreator) null);
      InputId inputId0 = nodeTraversal0.getInputId();
      assertNull(inputId0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("<");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      String[] stringArray0 = new String[1];
      JSError jSError0 = nodeTraversal0.makeError(node0, compiler0.OPTIMIZE_LOOP_ERROR, stringArray0);
      assertEquals((-1), jSError0.getCharno());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(";");
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
  public void test05()  throws Throwable  {
      ExpandJqueryAliases expandJqueryAliases0 = new ExpandJqueryAliases((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, expandJqueryAliases0);
      Node node0 = nodeTraversal0.getCurrentNode();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.common.collect.LinkeListMultimap$AsMapEntries");
      VarCheck varCheck0 = new VarCheck(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, varCheck0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse(node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      Compiler compiler1 = nodeTraversal0.getCompiler();
      assertSame(compiler0, compiler1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      compiler0.parseTestCode("com.google.common.codlect.LinkedListMultiqap$AsMapEntries");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "com.google.common.codlect.LinkedListMultiqap$AsMapEntries");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0, (ScopeCreator) null);
      JSModule jSModule0 = nodeTraversal0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      Node node0 = new Node(3);
      CheckLevel checkLevel0 = CheckLevel.OFF;
      JSError jSError0 = nodeTraversal0.makeError(node0, checkLevel0, compiler0.OPTIMIZE_LOOP_ERROR, (String[]) null);
      assertEquals((-1), jSError0.lineNumber);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("JSC_:ODE_TRAVERSAL_ERROR");
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
  public void test11()  throws Throwable  {
      ExpandJqueryAliases expandJqueryAliases0 = new ExpandJqueryAliases((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, expandJqueryAliases0);
      Node node0 = Node.newNumber((double) 2);
      AstParallelizer astParallelizer0 = AstParallelizer.createNewFunctionLevelAstParallelizer(node0, true);
      List<Node> list0 = astParallelizer0.split();
      // Undeclared exception!
      try { 
        nodeTraversal0.traverseRoots(list0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode(") != map(");
      ReferenceCollectingCallback.Behavior referenceCollectingCallback_Behavior0 = ReferenceCollectingCallback.DO_NOTHING_BEHAVIOR;
      ReferenceCollectingCallback referenceCollectingCallback0 = new ReferenceCollectingCallback(compiler0, referenceCollectingCallback_Behavior0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, referenceCollectingCallback0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse(node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      ArrayList<Node> arrayList0 = new ArrayList<Node>();
      nodeTraversal0.traverseRoots((List<Node>) arrayList0);
      assertTrue(arrayList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverseInnerNode(node0, node0, (Scope) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("long");
      ExternExportsPass externExportsPass0 = new ExternExportsPass(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, externExportsPass0);
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
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("FIT2]SdubK7^N2R");
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "FIT2]SdubK7^N2R");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      JSTypeNative jSTypeNative0 = JSTypeNative.ERROR_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      Scope scope0 = new Scope(node0, functionType0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("FIT2dubK7^N2R");
      FindExportableNodes findExportableNodes0 = new FindExportableNodes(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, findExportableNodes0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      JSTypeNative jSTypeNative0 = JSTypeNative.ERROR_FUNCTION_TYPE;
      FunctionType functionType0 = jSTypeRegistry0.getNativeFunctionType(jSTypeNative0);
      Scope scope0 = new Scope(node0, functionType0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      Node node0 = nodeTraversal0.getEnclosingFunction();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse((Node) null);
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
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.rhino.Node$ObjectPropListItem");
      MakeDeclaredNamesUnique makeDeclaredNamesUnique0 = new MakeDeclaredNamesUnique();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, makeDeclaredNamesUnique0);
      Scope scope0 = new Scope(node0, compiler0);
      nodeTraversal0.traverseAtScope(scope0);
      assertEquals(1, nodeTraversal0.getLineNumber());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      // Undeclared exception!
      try { 
        Normalize.parseAndNormalizeTestCode(compiler0, "O~{", "O~{");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, false);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "FIT2]SdubK7^N2R");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      boolean boolean0 = nodeTraversal0.hasScope();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, (NodeTraversal.Callback) null);
      Node node0 = Node.newString(132, "isNumber", 132, 132);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse(node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}