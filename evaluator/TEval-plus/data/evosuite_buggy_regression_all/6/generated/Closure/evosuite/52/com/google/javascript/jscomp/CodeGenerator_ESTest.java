/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:56:26 GMT 2023
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
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.IN_FOR_INIT_CLAUSE;
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = Node.newString(51, "-", (-1265), 1150);
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
  public void test01()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = Node.newString(")E&cc*Lcy5-&;X<");
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
  public void test02()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("_]>qVSOHkvB-/Y");
      assertEquals("/_]>qVSOHkvB-/Y/", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Node node0 = new Node(3257);
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown precedence for <unknown=3257> (type 3257)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
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
  public void test05()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
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
  public void test06()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      codeGenerator0.addList((Node) null, true);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Charset charset0 = Charset.forName("default");
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, charset0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, charset0);
      codeGenerator0.addAllSiblings((Node) null);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("2");
      assertEquals(2.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("\"dx</script~c(5mlb\"");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("default");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Node node0 = Node.newNumber(618.700502561544);
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addExpr(node0, 51);
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
      Node node0 = Node.newString(")E&cc*Lcy5-&;X<");
      CodeGenerator.Context codeGenerator_Context0 = CodeGenerator.Context.IN_FOR_INIT_CLAUSE;
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
  public void test15()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      codeGenerator0.addArrayList((Node) null);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = Node.newString(")E&ct*Lcs5-;Xq<", (-690), (-690));
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
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addJsString("'!TzTnC>f&R");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addJsString("bbP^3\"Kw8Dhk(");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("com.google.javascript.jscomp.CodeGenerator", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("\t\n\u000B\f\r \u0085\u1680\u2028\u2029\u205F\u3000\u00A0\u180E\u202F");
      assertEquals("\"\\t\\n\\u000b\\u000c\\r \\u0085\\u1680\\u2028\\u2029\\u205f\\u3000\\u00a0\\u180e\\u202f\"", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("}Q$Ni's<!--~Ay,!C");
      assertEquals("/}Q$Ni's<\\!--~Ay,!C/", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("2xW0->FW;");
      assertEquals("/2xW0->FW;/", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape(">t*SGYc//!8z } Sb6");
      assertEquals("/>t*SGYc//!8z } Sb6/", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("!-->ph[");
      assertEquals("\"!--\\>ph[\"", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("not dec</script a constructor");
      assertEquals("/not dec<\\/script a constructor/", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("r<H_");
      assertEquals("\"r<H_\"", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      CharsetEncoder charsetEncoder0 = charset0.newEncoder();
      String string0 = CodeGenerator.regexpEscape("3zff.S@Dl2m,2", charsetEncoder0);
      assertEquals("/3zff.S@Dl2m,2/", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String string0 = CodeGenerator.identifierEscape("\t\u000B\r \u0085\u1680{\u2029\u205F\u00A0\u180E\u202F");
      assertEquals("\\u0009\\u000b\\u000d \\u0085\\u1680{\\u2029\\u205f\\u00a0\\u180e\\u202f", string0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      String string0 = CodeGenerator.identifierEscape(":");
      assertEquals(":", string0);
  }
}
