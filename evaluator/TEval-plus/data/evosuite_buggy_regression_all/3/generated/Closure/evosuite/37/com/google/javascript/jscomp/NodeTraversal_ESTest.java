/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:09:46 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.AstParallelizer;
import com.google.javascript.jscomp.CheckGlobalThis;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.CheckSideEffects;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ConstCheck;
import com.google.javascript.jscomp.Denormalize;
import com.google.javascript.jscomp.FieldCleanupPass;
import com.google.javascript.jscomp.FindExportableNodes;
import com.google.javascript.jscomp.GroupVariableDeclarations;
import com.google.javascript.jscomp.JSError;
import com.google.javascript.jscomp.JSModule;
import com.google.javascript.jscomp.MinimizeExitPoints;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.OptimizeArgumentsArray;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.ScopeCreator;
import com.google.javascript.jscomp.SyntacticScopeCreator;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.rhino.Node;
import com.google.javascript.rhino.SimpleErrorReporter;
import com.google.javascript.rhino.jstype.JSType;
import com.google.javascript.rhino.jstype.JSTypeRegistry;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NodeTraversal_ESTest extends NodeTraversal_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection((AbstractCompiler) null);
      Node[] nodeArray0 = new Node[6];
      // Undeclared exception!
      try { 
        NodeTraversal.traverseRoots((AbstractCompiler) null, (NodeTraversal.Callback) checkSideEffects_StripProtection0, nodeArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("xS_=JZ");
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0, "xS_=JZ");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      Scope scope0 = new Scope(node0, compiler0);
      nodeTraversal0.traverseAtScope(scope0);
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("): ");
      Node node1 = compiler0.parseSyntheticCode("=E`|mV", "=E`|mV");
      Scope scope0 = new Scope(node1, compiler0);
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope1 = typedScopeCreator0.createScope(node0, scope0);
      assertEquals(0, scope1.getVarCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      Node node0 = compiler0.parseSyntheticCode("V*!N+}+6}wJ|+WK~HC", "V*!N+}+6}wJ|+WK~HC");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      String[] stringArray0 = new String[0];
      JSError jSError0 = nodeTraversal0.makeError(node0, compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      assertEquals((-1), jSError0.getCharno());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("P");
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
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = nodeTraversal0.getCurrentNode();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("xS_=JZ");
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0, "xS_=JZ");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, optimizeArgumentsArray0);
      String[] stringArray0 = new String[0];
      nodeTraversal0.report(node0, compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      assertFalse(node0.isSetterDef());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      Compiler compiler1 = nodeTraversal0.getCompiler();
      assertSame(compiler0, compiler1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("xS_=JZ", "xS_=JZ");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "X?X<Q$z");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      String[] stringArray0 = new String[0];
      JSError jSError0 = nodeTraversal0.makeError(node0, checkLevel0, compiler0.MOTION_ITERATIONS_ERROR, stringArray0);
      assertEquals(0, jSError0.getNodeSourceOffset());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkSideEffects_StripProtection0);
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal((JSTypeRegistry) null, "msg.no.paren.parms");
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, (Node) null, (Node) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("xS_=JZ", "xS_=JZ");
      JSTypeRegistry jSTypeRegistry0 = compiler0.getTypeRegistry();
      FieldCleanupPass.QualifiedNameSearchTraversal fieldCleanupPass_QualifiedNameSearchTraversal0 = new FieldCleanupPass.QualifiedNameSearchTraversal(jSTypeRegistry0, "X?X<Q$z");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, fieldCleanupPass_QualifiedNameSearchTraversal0);
      boolean boolean0 = fieldCleanupPass_QualifiedNameSearchTraversal0.shouldTraverse(nodeTraversal0, node0, node0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("anchor");
      node0.addChildToBack(node0);
      Denormalize.StripConstantAnnotations denormalize_StripConstantAnnotations0 = new Denormalize.StripConstantAnnotations(compiler0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) denormalize_StripConstantAnnotations0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("");
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      FindExportableNodes findExportableNodes0 = new FindExportableNodes((AbstractCompiler) null);
      LinkedList<Node> linkedList0 = new LinkedList<Node>();
      NodeTraversal.traverseRoots((AbstractCompiler) null, (List<Node>) linkedList0, (NodeTraversal.Callback) findExportableNodes0);
      assertEquals(0, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      Node node0 = compiler0.parseTestCode("xS_=JZ");
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      AstParallelizer astParallelizer0 = AstParallelizer.createNewFileLevelAstParallelizer(node0);
      List<Node> list0 = astParallelizer0.split();
      // Undeclared exception!
      try { 
        nodeTraversal0.traverseRoots(list0);
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
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      Node node0 = compiler0.parseTestCode("xS_JZ");
      Scope scope0 = new Scope(node0, compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
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
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      Node node0 = compiler0.parseTestCode("");
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Scope scope0 = syntacticScopeCreator0.createScope(node0, (Scope) null);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(1, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator((AbstractCompiler) null);
      MinimizeExitPoints minimizeExitPoints0 = new MinimizeExitPoints((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, minimizeExitPoints0, syntacticScopeCreator0);
      SimpleErrorReporter simpleErrorReporter0 = new SimpleErrorReporter();
      JSTypeRegistry jSTypeRegistry0 = new JSTypeRegistry(simpleErrorReporter0, true);
      ArrayList<JSType> arrayList0 = new ArrayList<JSType>();
      Node node0 = jSTypeRegistry0.createParametersWithVarArgs((List<JSType>) arrayList0);
      nodeTraversal0.traverseInnerNode(node0, node0, (Scope) null);
      int int0 = nodeTraversal0.getLineNumber();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      Node node0 = compiler0.parseTestCode("");
      nodeTraversal0.traverse(node0);
      nodeTraversal0.getModule();
      assertEquals(1, nodeTraversal0.getLineNumber());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
      compiler0.parseTestCode("");
      JSModule jSModule0 = nodeTraversal0.getModule();
      assertNull(jSModule0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      Node node0 = nodeTraversal0.getEnclosingFunction();
      assertNull(node0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) null, (Node) null, (NodeTraversal.Callback) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeTraversal", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("`pf#r::F");
      SyntacticScopeCreator syntacticScopeCreator0 = new SyntacticScopeCreator(compiler0);
      Scope scope0 = syntacticScopeCreator0.createScope(node0, (Scope) null);
      GroupVariableDeclarations groupVariableDeclarations0 = new GroupVariableDeclarations(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, groupVariableDeclarations0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertEquals(" [testcode] ", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("xS_=JZ", "xS_=JZ");
      ConstCheck constCheck0 = new ConstCheck(compiler0);
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) constCheck0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0, (ScopeCreator) null);
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
  public void test25()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("xS_=JZ");
      OptimizeArgumentsArray optimizeArgumentsArray0 = new OptimizeArgumentsArray(compiler0, "xS_=JZ");
      // Undeclared exception!
      try { 
        NodeTraversal.traverse((AbstractCompiler) compiler0, node0, (NodeTraversal.Callback) optimizeArgumentsArray0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CheckSideEffects.StripProtection checkSideEffects_StripProtection0 = new CheckSideEffects.StripProtection((AbstractCompiler) null);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, checkSideEffects_StripProtection0);
      boolean boolean0 = nodeTraversal0.hasScope();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseSyntheticCode("xS_=JZ", "xS_=JZ");
      Node node1 = new Node(1, node0);
      node0.srcref(node1);
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
}
