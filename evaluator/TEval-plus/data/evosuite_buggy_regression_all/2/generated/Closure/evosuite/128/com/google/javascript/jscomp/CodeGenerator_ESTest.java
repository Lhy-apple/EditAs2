/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:37:29 GMT 2023
 */

package com.google.javascript.jscomp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import com.google.javascript.jscomp.AliasKeywords;
import com.google.javascript.jscomp.CodeConsumer;
import com.google.javascript.jscomp.CodeGenerator;
import com.google.javascript.jscomp.CompilerOptions;
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
      int int0 = AliasKeywords.estimateMinOccurrencesRequriedToAlias();
      assertEquals(14, int0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.regexpEscape("3j><!--C+`");
      assertEquals("/3j>\\x3c!--C+`/", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.tagAsStrict();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addCaseBody((Node) null);
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
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.regexpEscape("yE>zZnI/|=");
      assertEquals("/yE>zZnI/|\\u007f=/", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Node node0 = Node.newString(2092, "var", 2092, 2092);
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, false);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 2092
         //
         verifyException("com.google.javascript.rhino.Token", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("0");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      boolean boolean0 = CodeGenerator.isSimpleNumber(",drj+%KF");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      boolean boolean0 = CodeGenerator.isSimpleNumber("1m(");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      boolean boolean0 = CodeGenerator.isSimpleNumber("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("1");
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Node node0 = new Node(85);
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addArrayList(node0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = Node.newString(":Gn,%`u-vd_");
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.IN_FOR_INIT_CLAUSE;
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
  public void test13()  throws Throwable  {
      Node node0 = new Node(51);
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.IN_FOR_INIT_CLAUSE;
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
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
  public void test14()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      codeGenerator0.addArrayList((Node) null);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      codeGenerator0.addAllSiblings((Node) null);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Node node0 = new Node(4, 4, 4);
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
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
  public void test17()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.regexpEscape("l>\n");
      assertEquals("/l>\\n/", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("{}a \"U[u}S=U{V");
      assertEquals("\"{}a \\\"U[u}S\\x3dU{V\"", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("0,T[4oqU<685T&(");
      assertEquals("\"0,T[4oqU\\x3c685T\\x26(\"", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.regexpEscape("e`EM)+[/Qq8nPNl('U!");
      assertEquals("/e`EM)+[/Qq8nPNl('U!/", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.regexpEscape("-=?)pA2\"xLURd");
      assertEquals("/-=?)pA2\"xLURd/", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("V|hxQY&");
      assertEquals("\"V|hxQY&\"", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.regexpEscape("V|hxQY&");
      assertEquals("/V|hxQY&/", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      CompilerOptions compilerOptions0 = new CompilerOptions();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, compilerOptions0);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString("#<\"c,u>");
      assertEquals("\"#\\x3c\\\"c,u\\x3e\"", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      Charset charset0 = Charset.defaultCharset();
      CharsetEncoder charsetEncoder0 = charset0.newEncoder();
      String string0 = codeGenerator0.regexpEscape("U5@>->", charsetEncoder0);
      assertEquals("/U5@>->/", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString(";VGQF;(_<!--gAa");
      assertEquals("\";VGQF;(_\\x3c!--gAa\"", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.escapeToDoubleQuotedJsString(";vgqf;_</scriptaa");
      assertEquals("\";vgqf;_\\x3c/scriptaa\"", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      CodeGenerator codeGenerator0 = CodeGenerator.forCostEstimation((CodeConsumer) null);
      String string0 = codeGenerator0.regexpEscape("PN3`NPav9<*m");
      assertEquals("/PN3`NPav9<*m/", string0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      String string0 = CodeGenerator.identifierEscape("0\u0660\u06F0\u07C0\u0966\u09E6\u0A66\u0AE6\u0B66\u0BE6\u0C66\u0CE6\u0D66\u0E50\u0ED0\u0F20\u1040\u1090\u17E0\u1810\u1946\u19D0\u1B50$\u1C40\u1C50\uA620\uA8D0\uA900/\uFF10");
      assertEquals("0\\u0660\\u06f0\\u07c0\\u0966\\u09e6\\u0a66\\u0ae6\\u0b66\\u0be6\\u0c66\\u0ce6\\u0d66\\u0e50\\u0ed0\\u0f20\\u1040\\u1090\\u17e0\\u1810\\u1946\\u19d0\\u1b50$\\u1c40\\u1c50\\ua620\\ua8d0\\ua900/\\uff10", string0);
  }
}