/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:10:50 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NameAnalyzer;
import com.google.javascript.jscomp.Normalize;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NameAnalyzer_ESTest extends NameAnalyzer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "window", "window");
      nameAnalyzer0.process(node0, node0);
      assertEquals(31, Node.SIDE_EFFECTS_FLAGS_MASK);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(499);
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      nameAnalyzer0.process(node0, node0);
      String string0 = nameAnalyzer0.getHtmlReport();
      assertEquals("<html><body><style type=\"text/css\">body, td, p {font-family: Arial; font-size: 83%} ul {margin-top:2px; margin-left:0px; padding-left:1em;} li {margin-top:3px; margin-left:24px; padding-left:0px;padding-bottom: 4px}</style>OVERALL STATS<ul><li>Total Names: 2</li>\n<li>Total Classes: 0</li>\n<li>Total Static Functions: 2</li>\n<li>Referenced Names: 2</li>\n<li>Referenced Classes: 0</li>\n<li>Referenced Functions: 2</li>\n</ul>ALL NAMES<ul>\n<li><a name=\"Function\">Function</a><ul></li></ul></li><li><a name=\"window\">window</a><ul></li></ul></li></ul></body></html>", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "Xu2=bs", "Xu2=bs");
      nameAnalyzer0.process(node0, node0);
      assertEquals(0, Node.SIDE_EFFECTS_ALL);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = new Node(147);
      // Undeclared exception!
      try { 
        nameAnalyzer0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = Normalize.parseAndNormalizeTestCode(compiler0, "Xss", "Xss");
      Node node1 = new Node(37, node0, 49, 16);
      nameAnalyzer0.process(node0, node1);
      assertEquals(0, node0.getCharno());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.NameAnalyzer$ProcessExternals");
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      nameAnalyzer0.process(node0, node0);
      assertFalse(node0.isUnscopedQualifiedName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = compiler0.parseTestCode("com.google.javascript.jscomp.Na+eAnalyzer$ProcessExternals");
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node1 = new Node(118, node0, node0);
      // Undeclared exception!
      try { 
        nameAnalyzer0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // INTERNAL COMPILER ERROR.
         // Please report this problem.
         // null
         //   Node(NAME com):  [testcode] :1:0
         // com.google.javascript.jscomp.Na+eAnalyzer$ProcessExternals
         //   Parent(GETPROP):  [testcode] :1:0
         // com.google.javascript.jscomp.Na+eAnalyzer$ProcessExternals
         //
         verifyException("com.google.common.base.Preconditions", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = new Node((-1722485283));
      Node node1 = new Node(118, node0, node0);
      nameAnalyzer0.process(node0, node1);
      assertFalse(node0.isVarArgs());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node(115, 115, 115);
      Node node1 = new Node(36, node0, 37, 16);
      // Undeclared exception!
      try { 
        nameAnalyzer0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = new Node(38);
      Node node1 = new Node(49, node0, node0, node0);
      // Undeclared exception!
      try { 
        nameAnalyzer0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = new Node((-1722485307));
      Node node1 = new Node(108, node0, node0);
      nameAnalyzer0.process(node1, node1);
      assertEquals(49, Node.DIRECT_EVAL);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = new Node((-1722485307));
      Node node1 = new Node(113, node0, node0);
      nameAnalyzer0.process(node1, node1);
      assertEquals(38, Node.SYNTHETIC_BLOCK_PROP);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node(38, 38, 38);
      Node node1 = new Node(119, node0, node0);
      // Undeclared exception!
      try { 
        nameAnalyzer0.process(node0, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node((-1722485306));
      Node node1 = new Node(110, node0, node0);
      nameAnalyzer0.process(node1, node1);
      assertFalse(node1.isNull());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node((-4390));
      Node node1 = new Node(114, node0, 36, 30);
      nameAnalyzer0.process(node0, node1);
      assertFalse(node1.isContinue());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node(1011);
      nameAnalyzer0.process(node0, node0);
      nameAnalyzer0.process(node0, node0);
      assertEquals(41, Node.BRACELESS_TYPE);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node((-25));
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node1 = new Node(35, node0, 39, 40);
      Node node2 = new Node(4, node1, 4, 1);
      nameAnalyzer0.process(node1, node2);
      assertFalse(node1.isDo());
  }
}
