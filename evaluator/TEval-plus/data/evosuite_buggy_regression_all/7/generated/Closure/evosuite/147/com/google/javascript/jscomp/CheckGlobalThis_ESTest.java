/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:21:46 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AbstractCompiler;
import com.google.javascript.jscomp.CheckGlobalThis;
import com.google.javascript.jscomp.CheckLevel;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NodeTraversal;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckGlobalThis_ESTest extends CheckGlobalThis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(105);
      Node node1 = new Node(30, node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      Node node0 = new Node(105);
      Node node1 = new Node(125, node0, node0, node0, (-2), 1);
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node0 = new Node(105);
      Node node1 = new Node(12, node0, node0, node0, 0, 14);
      Node node2 = compiler0.parseSyntheticCode("", "");
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node2);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(105);
      Node node1 = new Node(38, node0, node0, node0, node0);
      Node node2 = new Node(30, node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(105);
      Node node1 = new Node(86);
      Node node2 = new Node(12, node1, node1, node0, 8202, 83);
      // Undeclared exception!
      try { 
        checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CheckGlobalThis", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(100);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, (Node) null);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node((-1433));
      Node node1 = new Node(86, node0, node0, node0, 9, (-1433));
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node((-1433));
      Node node1 = new Node(86, node0, node0, node0, 9, (-1433));
      checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      checkGlobalThis0.visit((NodeTraversal) null, node0, node0);
      assertEquals(1, Node.SPECIALCALL_EVAL);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node((-1448));
      Node node1 = new Node(86, node0, node0, node0, 9, (-1448));
      checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(105);
      checkGlobalThis0.visit((NodeTraversal) null, node0, node0);
      assertEquals(2, Node.ATTRIBUTE_FLAG);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(42);
      checkGlobalThis0.visit((NodeTraversal) null, node0, node0);
      assertEquals(11, Node.USES_PROP);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newString(42, "next() has not been called");
      checkGlobalThis0.visit((NodeTraversal) null, node0, (Node) null);
      assertEquals(46, Node.IS_NAMESPACE);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = new Node(105);
      Node node1 = new Node(86, node0, node0, node0, 41, 12);
      NodeTraversal nodeTraversal0 = new NodeTraversal(compiler0, checkGlobalThis0);
      // Undeclared exception!
      try { 
        nodeTraversal0.traverse(node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}