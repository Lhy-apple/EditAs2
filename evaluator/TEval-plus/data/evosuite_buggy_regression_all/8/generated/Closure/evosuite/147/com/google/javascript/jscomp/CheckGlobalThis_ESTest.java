/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:29:55 GMT 2023
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
import java.io.PrintStream;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CheckGlobalThis_ESTest extends CheckGlobalThis_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node0 = Node.newString(42, "w(", 42, 42);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, checkGlobalThis0);
      nodeTraversal0.traverse(node0);
      assertEquals(29, Node.JSDOC_INFO_PROP);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Node node0 = Node.newString(105, "Cw(");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      node0.addChildToFront(node0);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Node node0 = Node.newString(105, "w(", 96, 96);
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node1 = new Node(125, node0, 7, 1285);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Node node0 = Node.newString(105, "Cw`", 151, 151);
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node1 = new Node(154, node0, 39, 40);
      Compiler compiler0 = new Compiler();
      Node node2 = compiler0.parseTestCode("prototype");
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node2);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Node node0 = Node.newString(105, "Cw(");
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node1 = new Node(38, node0, 44, 1);
      node0.addChildToFront(node1);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, checkGlobalThis0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse(nodeTraversal0, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = Node.newString(105, "w(", 105, 105);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node1 = new Node(86, node0, node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newString(82, "%cYFjr=+=iVheWo |g", 82, (-1016));
      Node node1 = Node.newString(86, "%cYFjr=+=iVheWo |g", 36, 3633);
      node1.addChildToFront(node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      boolean boolean1 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      assertTrue(boolean1 == boolean0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Node node0 = Node.newString(105, "w(", 105, 105);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node1 = new Node(86, node0, node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newString(35, "%cYFjr=+=iVheWo |g", 35, 35);
      Node node1 = Node.newString(86, "%cYFjr=+=iVheWo |g", 36, 2);
      node1.addChildToFront(node0);
      // Undeclared exception!
      try { 
        checkGlobalThis0.shouldTraverse((NodeTraversal) null, node1, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CheckGlobalThis", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      CheckLevel checkLevel0 = CheckLevel.OFF;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newString(33, ">S$RO* ,g cj5XDH", 33, 33);
      Node node1 = Node.newString(86, ">S$RO* ,g cj5XDH", 36, 2);
      node0.addChildrenToBack(node1);
      node1.addChildToFront(node0);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node1, node1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      CheckLevel checkLevel0 = CheckLevel.WARNING;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newString(33, "&=", 33, 125);
      Node node1 = Node.newString(86, "prototype", (-717), 35);
      node1.addChildrenToBack(node0);
      node0.addChildToFront(node1);
      boolean boolean0 = checkGlobalThis0.shouldTraverse((NodeTraversal) null, node1, node1);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node0 = Node.newString(42, "nb+bW]UD", (-1453), 106);
      NodeTraversal nodeTraversal0 = new NodeTraversal((AbstractCompiler) null, checkGlobalThis0);
      Node node1 = Node.newString(35, "DJ*PN;5EB", 32, 16);
      // Undeclared exception!
      try { 
        checkGlobalThis0.visit(nodeTraversal0, node0, node1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CheckGlobalThis", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler((PrintStream) null);
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis(compiler0, checkLevel0);
      Node node0 = Node.newString(82, "%cYFjr=+=iVheWo |g", 82, (-1016));
      Node node1 = Node.newString(86, "%cYFjr=+=iVheWo |g", 36, 3633);
      node1.addChildToFront(node0);
      checkGlobalThis0.shouldTraverse((NodeTraversal) null, node0, node1);
      checkGlobalThis0.visit((NodeTraversal) null, node0, node0);
      assertEquals(7, Node.LOCAL_PROP);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CheckLevel checkLevel0 = CheckLevel.ERROR;
      CheckGlobalThis checkGlobalThis0 = new CheckGlobalThis((AbstractCompiler) null, checkLevel0);
      Node node0 = Node.newString(42, "nb+bW]UD", (-1453), 106);
      checkGlobalThis0.visit((NodeTraversal) null, node0, node0);
      assertEquals(35, Node.QUOTED_PROP);
  }
}