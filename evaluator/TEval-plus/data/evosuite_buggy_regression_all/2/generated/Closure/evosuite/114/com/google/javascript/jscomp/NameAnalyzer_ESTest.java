/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:36:22 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.Compiler;
import com.google.javascript.jscomp.NameAnalyzer;
import com.google.javascript.rhino.Node;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NameAnalyzer_ESTest extends NameAnalyzer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = compiler0.parseSyntheticCode("window", "window");
      nameAnalyzer0.process(node0, node0);
      assertFalse(node0.isGetProp());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node((-298), (-298), (-298));
      nameAnalyzer0.process(node0, node0);
      String string0 = nameAnalyzer0.getHtmlReport();
      assertEquals("<html><body><style type=\"text/css\">body, td, p {font-family: Arial; font-size: 83%} ul {margin-top:2px; margin-left:0px; padding-left:1em;} li {margin-top:3px; margin-left:24px; padding-left:0px;padding-bottom: 4px}</style>OVERALL STATS<ul><li>Total Names: 2</li>\n<li>Total Classes: 0</li>\n<li>Total Static Functions: 2</li>\n<li>Referenced Names: 2</li>\n<li>Referenced Classes: 0</li>\n<li>Referenced Functions: 2</li>\n</ul>ALL NAMES<ul>\n<li><a name=\"Function\">Function</a><ul></li></ul></li><li><a name=\"window\">window</a><ul></li></ul></li></ul></body></html>", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node(65536, 65536, 65536);
      Node node1 = new Node(86, node0, 1, (-138));
      // Undeclared exception!
      try { 
        nameAnalyzer0.process(node1, node1);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = new Node(147);
      // Undeclared exception!
      try { 
        nameAnalyzer0.process(node0, node0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = compiler0.parseSyntheticCode("com.google.common.collect.Ordering", "nw$");
      Node node1 = new Node(114, node0, 43, 51);
      nameAnalyzer0.process(node1, node1);
      assertFalse(node1.isNew());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      Node node0 = new Node(65536, (-1), (-1));
      Node node1 = new Node(4, node0, node0, 2, 2);
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      nameAnalyzer0.process(node0, node1);
      assertEquals(51, Node.STATIC_SOURCE_FILE);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = new Node(1, 1, 1);
      Node node1 = new Node(49, node0, (-2428), 8);
      nameAnalyzer0.process(node1, node1);
      assertFalse(node1.isSetterDef());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, false);
      Node node0 = compiler0.parseSyntheticCode("com.google.javascript.jscomp.PeepholeOptimizationsPass$PepCallback", "com.google.javascript.jscomp.PeepholeOptimizationsPass$PepCallback");
      nameAnalyzer0.process(node0, node0);
      assertFalse(node0.isWith());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      Compiler compiler0 = new Compiler();
      NameAnalyzer nameAnalyzer0 = new NameAnalyzer(compiler0, true);
      Node node0 = compiler0.parseSyntheticCode("com.google.common.collect.Ordering", "nw$");
      nameAnalyzer0.process(node0, node0);
      nameAnalyzer0.process(node0, node0);
      assertEquals(0, node0.getCharno());
  }
}
