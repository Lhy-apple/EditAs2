/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:44:14 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.CodeConsumer;
import com.google.javascript.jscomp.CodeGenerator;
import com.google.javascript.rhino.Node;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CodeGenerator_ESTest extends CodeGenerator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Charset charset0 = Charset.forName("default");
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, charset0);
      // Undeclared exception!
      try { 
        codeGenerator0.add("F+,;5eR]Xq8FrI");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = Node.newString("/GTc+LrgfODKj<|5w");
      int[] intArray0 = new int[0];
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, intArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("/@&7-]>O>");
      assertEquals("\"/@&7-]>O>\"", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = new Node(0);
      // Undeclared exception!
      try { 
        codeGenerator0.addCaseBody(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = Node.newNumber(2479.866844897);
      // Undeclared exception!
      try { 
        codeGenerator0.addAllSiblings(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = new Node(24);
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, true);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, charset0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = Node.newString("/scrppt");
      // Undeclared exception!
      try { 
        codeGenerator0.addExpr(node0, 46);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Node node0 = Node.newNumber(3597.192353284951);
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.IN_FOR_INIT_CLAUSE;
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, true, codeGenerator_Context0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      codeGenerator0.addList((Node) null);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Node node0 = Node.newNumber(3615.519083892013);
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.OTHER;
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, false, codeGenerator_Context0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      codeGenerator0.addList((Node) null, (int[]) null);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = Node.newString("/scrppt");
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, (int[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = new Node(835);
      int[] intArray0 = new int[1];
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, intArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = new Node(835);
      int[] intArray0 = new int[1];
      intArray0[0] = 49;
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, intArray0);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown precedence for <unknown=835> (type 835)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      codeGenerator0.addAllSiblings((Node) null);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      String string0 = CodeGenerator.jsString("/@'&Z,7->O>", (CharsetEncoder) null);
      assertEquals("\"/@'&Z,7->O>\"", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      String string0 = CodeGenerator.jsString("8z)@COGq)\rnA?-\":P", (CharsetEncoder) null);
      assertEquals("'8z)@COGq)\\rnA?-\":P'", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("\n");
      assertEquals("/\\n/", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      CharsetEncoder charsetEncoder0 = charset0.newEncoder();
      String string0 = CodeGenerator.jsString("nyek0#lm6</scriptos", charsetEncoder0);
      assertEquals("\"nyek0#lm6<\\/scriptos\"", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("w>e{6;l");
      assertEquals("\"w>e{6;l\"", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("kP>KO|[]]>");
      assertEquals("\"kP>KO|[]]\\>\"", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("fTr<>E{Q;lj}");
      assertEquals("/fTr<>E{Q;lj}/", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      String string0 = CodeGenerator.identifierEscape("default");
      assertEquals("default", string0);
  }
}
