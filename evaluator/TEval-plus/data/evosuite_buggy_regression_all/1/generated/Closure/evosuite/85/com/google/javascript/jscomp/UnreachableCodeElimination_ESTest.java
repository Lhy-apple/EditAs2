/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:11:27 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.ControlFlowGraph;
import com.google.javascript.jscomp.ConvertToDottedProperties;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.jscomp.Scope;
import com.google.javascript.jscomp.TypedScopeCreator;
import com.google.javascript.jscomp.UnreachableCodeElimination;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class UnreachableCodeElimination_ESTest extends UnreachableCodeElimination_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, true);
      // Undeclared exception!
      try { 
        unreachableCodeElimination0.process((Node) null, (Node) null);
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
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      unreachableCodeElimination0.visit(nodeTraversal0, (Node) null, (Node) null);
      assertEquals("", nodeTraversal0.getSourceName());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(105);
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, false);
      ConvertToDottedProperties convertToDottedProperties0 = new ConvertToDottedProperties(compiler0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, convertToDottedProperties0);
      unreachableCodeElimination0.visit(nodeTraversal0, node0, node0);
      assertEquals(46, Node.IS_DISPATCHER);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, false);
      Node node0 = compiler0.parseSyntheticCode("@m$\"r`/na9x@", "@m$\"r`/na9x@");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      Node node1 = new Node(114, node0, node0, node0, node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      nodeTraversal0.traverseInnerNode(node1, node0, scope0);
      assertFalse(node1.isQualifiedName());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, true);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      Node node0 = Node.newString("FdWXHC`Uw .mE#{+A)");
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      unreachableCodeElimination0.curCfg = controlFlowGraph0;
      // Undeclared exception!
      try { 
        unreachableCodeElimination0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, true);
      Node node0 = compiler0.parseSyntheticCode("S", "ghdvE1Qo>g*]dNF|~");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertEquals(0, node0.getChildCount());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, true);
      compiler0.parseSyntheticCode("S", "ghdvE1Qo>g*]dNF|~");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Node node0 = new Node(0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertFalse(node0.isOnlyModifiesThisCall());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, false);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.UnreachableCodeElimination", "com.google.javascript.jscomp.UnreachableCodeElimination");
      TypedScopeCreator typedScopeCreator0 = new TypedScopeCreator(compiler0);
      Scope scope0 = typedScopeCreator0.createInitialScope(node0);
      nodeTraversal0.traverseInnerNode(node0, node0, scope0);
      assertTrue(node0.hasChildren());
      assertTrue(node0.hasOneChild());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      UnreachableCodeElimination unreachableCodeElimination0 = new UnreachableCodeElimination(compiler0, true);
      Node node0 = compiler0.parseSyntheticCode("C`6.{;[Mv1@(%", "C`6.{;[Mv1@(%");
      node0.addChildToFront(node0);
      ControlFlowGraph<Node> controlFlowGraph0 = new ControlFlowGraph<Node>(node0, true, true);
      unreachableCodeElimination0.curCfg = controlFlowGraph0;
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, unreachableCodeElimination0);
      // Undeclared exception!
      try { 
        unreachableCodeElimination0.visit(nodeTraversal0, node0, node0);
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }
}