/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:05:29 GMT 2023
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
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addJsString("E_DOUBLE\u0010\u0001\u0012\u000E\n\nTYPE_FLOAT\u0010\u0002\u0012\u000E\n\nTYPE_INT64\u0010\u0003\u0012\u000F\n\u000BTYPE_UINT64\u0010\u0004\u0012\u000E\n\nTYPE_INT32\u0010\u0005\u0012\u0010\n\fTYPE_FIXED64\u0010\u0006\u0012\u0010\n\fTYPE_FIXED32\u0010\u0007\u0012\r\n\tTYPE_BOOL\u0010\b\u0012\u000F\n\u000BTYPE_STRING\u0010\t\u0012\u000E\n\nTYPE_GROUP\u0010\n\u0012\u0010\n\fTYPE_MESSAGE\u0010\u000B\u0012\u000E\n\nTYPE_BYTES\u0010\f\u0012\u000F\n\u000BTYPE_UINT32\u0010\r\u0012\r\n\tTYPE_ENUM\u0010\u000E\u0012\u0011\n\rTYPE_SFIXED32\u0010\u000F\u0012\u0011\n\rTYPE_SFIXED64\u0010\u0010\u0012\u000F\n\u000BTYPE_SINT32\u0010\u0011\u0012\u000F\n\u000BTYPE_SINT64\u0010\u0012\"C\n\u0005Label\u0012\u0012\n\u000ELABEL_OPTIONAL\u0010\u0001\u0012\u0012\n\u000ELABEL_REQUIRED\u0010\u0002\u0012\u0012\n\u000ELABEL_REPEATED\u0010\u0003\"\u008C\u0001\n\u0013EnumDescriptorProto\u0012\f\n\u0004name\u0018\u0001");
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
      String string0 = CodeGenerator.regexpEscape("E_DOUBLE\u0010\u0001\u0012\u000E\n\nTYPE_FLOAT\u0010\u0002\u0012\u000E\n\nTYPE_INT64\u0010\u0003\u0012\u000F\n\u000BTYPE_UINT64|\u0004\u0012\u000E\n\nTYPE_INT32\u0010\u0005\u0012\u0010\n\fTYPE_FIXED64\u0010\u0006\u0012\u0010\n\fTYPE_FIXED32\u0010\u0007\u0012\r\n\tTYPE_BOOL\u0010\b\u0012\u000F\n\u000BTYPE_STRI-G\u0010\t\u0012\u000E\n\nTYE_GROUP\u0010)\u0012\u0010\n\fTYPE_MESSAGE\u0010\u000B\u00125\n\nTYPE_BYTE7\u0010\f\u0012\u000F\n\u000BTYPE_UINT32\u0010\r\u0012\r\n\tTYPE_ENUM\u0010\u000E\u0012\u0011\n\rTYPE_SFIXED32\u0010\u000F\u0012\u0011\n\rTYPE_SFIXED64\u0010\u0010\u0012\u000F\n\u000BTYPE_SINT32\u0010\u0011\u0012\u000F\n\u000BTYPE_SINTb4\u0010\u0012\"C\n\u0005Label\u0012\u0012\n\u000ELABEL_OPTIONAL\u0010\u0001\u0012\u0012\n\u000ELABEL_REQUIRED\u0010\u0002\u0012\u0012\n\u000ELABEL_REPEATED\u0010\u0003\"\u008C\u0001\n\u0013EnumDescriptorProto\u0012\f\n\u0004name\u0018\u0001");
      assertEquals("/E_DOUBLE\\u0010\\u0001\\u0012\\u000e\\n\\nTYPE_FLOAT\\u0010\\u0002\\u0012\\u000e\\n\\nTYPE_INT64\\u0010\\u0003\\u0012\\u000f\\n\\u000bTYPE_UINT64|\\u0004\\u0012\\u000e\\n\\nTYPE_INT32\\u0010\\u0005\\u0012\\u0010\\n\\u000cTYPE_FIXED64\\u0010\\u0006\\u0012\\u0010\\n\\u000cTYPE_FIXED32\\u0010\\u0007\\u0012\\r\\n\\tTYPE_BOOL\\u0010\\u0008\\u0012\\u000f\\n\\u000bTYPE_STRI-G\\u0010\\t\\u0012\\u000e\\n\\nTYE_GROUP\\u0010)\\u0012\\u0010\\n\\u000cTYPE_MESSAGE\\u0010\\u000b\\u00125\\n\\nTYPE_BYTE7\\u0010\\u000c\\u0012\\u000f\\n\\u000bTYPE_UINT32\\u0010\\r\\u0012\\r\\n\\tTYPE_ENUM\\u0010\\u000e\\u0012\\u0011\\n\\rTYPE_SFIXED32\\u0010\\u000f\\u0012\\u0011\\n\\rTYPE_SFIXED64\\u0010\\u0010\\u0012\\u000f\\n\\u000bTYPE_SINT32\\u0010\\u0011\\u0012\\u000f\\n\\u000bTYPE_SINTb4\\u0010\\u0012\"C\\n\\u0005Label\\u0012\\u0012\\n\\u000eLABEL_OPTIONAL\\u0010\\u0001\\u0012\\u0012\\n\\u000eLABEL_REQUIRED\\u0010\\u0002\\u0012\\u0012\\n\\u000eLABEL_REPEATED\\u0010\\u0003\"\\u008c\\u0001\\n\\u0013EnumDescriptorProto\\u0012\\u000c\\n\\u0004name\\u0018\\u0001/", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Node node0 = Node.newString("com.google.protobuf.DescriptorProtos$UninterpretedOption$NamePart$Builder");
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0);
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
  public void test04()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("z^@N]>x^eMyZ=XC ");
      assertEquals("\"z^@N]>x^eMyZ=XC \"", string0);
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
      Node node0 = Node.newNumber(158.6);
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
  public void test07()  throws Throwable  {
      Node node0 = new Node((-2147483646));
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      // Undeclared exception!
      try { 
        codeGenerator0.addList(node0, false);
        fail("Expecting exception: Error");
      
      } catch(Error e) {
         //
         // Unknown precedence for <unknown=-2147483646> (type -2147483646)
         //
         verifyException("com.google.javascript.jscomp.NodeUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Charset charset0 = Charset.forName("default");
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, charset0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("0");
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("#\"4>GJ;8aC;Qh");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double double0 = CodeGenerator.getSimpleNumber("");
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Node node0 = new Node(85);
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
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
  public void test13()  throws Throwable  {
      Node node0 = Node.newNumber((-1666.18));
      Charset charset0 = Charset.defaultCharset();
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null, charset0);
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
  public void test14()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      Node node0 = new Node(51);
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
      codeGenerator0.addList((Node) null);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      codeGenerator0.addArrayList((Node) null);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      codeGenerator0.addAllSiblings((Node) null);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CodeGenerator codeGenerator0 = new CodeGenerator((CodeConsumer) null);
      String string0 = codeGenerator0.jsString("('Pe4\"AvE,)Vgmb'");
      assertEquals("\"('Pe4\\\"AvE,)Vgmb'\"", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("/b&toy/$</script");
      assertEquals("//b&toy/$<\\/script/", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape("<");
      assertEquals("/</", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      String string0 = CodeGenerator.regexpEscape(">>");
      assertEquals("/>>/", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("!-u->pzi");
      assertEquals("\"!-u->pzi\"", string0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("!-->pzi");
      assertEquals("\"!--\\>pzi\"", string0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      String string0 = CodeGenerator.escapeToDoubleQuotedJsString("dV5<!--6Y");
      assertEquals("\"dV5\\u007f<\\!--6Y\"", string0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Charset charset0 = Charset.defaultCharset();
      CharsetEncoder charsetEncoder0 = charset0.newEncoder();
      String string0 = CodeGenerator.regexpEscape("H[C", charsetEncoder0);
      assertEquals("/H[C/", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      String string0 = CodeGenerator.identifierEscape("E_DOUBLE\u0010\u0001\u0012\u000E\n\nTYPE_FLOAT\u0010\u0002\u0012\u000E\n\nTYPE_INT64\u0010\u0003\u0012\u000F\n\u000BTYPE_UINT64|\u0004\u0012\u000E\n\nTYPE_INT32\u0010\u0005\u0012\u0010\n\fTYPE_FIXED64\u0010\u0006\u0012\u0010\n\fTYPE_FIXED32\u0010\u0007\u0012\r\n\tTYPE_BOOL\u0010\b\u0012\u000F\n\u000BTYPE_STRI-G\u0010\t\u0012\u000E\n\nTYE_GROUP\u0010)\u0012\u0010\n\fTYPE_MExSAGE\u0010\u000B\u00125\n\nTYPE_BYTE7\u0010\f\u0012\u000F\n\u000BTYPE_UINT32\u0010\r\u0012\r\n\tTYPE_ENUM\u0010\u000E\u0012\u0011\n\rTYPE_SFIXED32\u0010\u000F\u0012\u0011\n\rTYPE_SFIXED64\u0010\u0010\u0012\u000F\n\u000BTYPE_SINT32E\u0011\u0012\u000F\n\u000BTYPE_SINTb4\u0010\u0012\"C\n\u0005Labe \u0012\u0012\n\u000ELABEL_OPTIONAL\u0010\u0001\u0012\u0012\n\u000ELABEL_REQUIRED\u0010\u0002\u0012\u0012\n\u000ELABEL_REPEATED\u0010\u0003\"\u008C\u0001\n\u0013EnumDescriptorProto\u0012\f\n\u0004name\u0018\u0001");
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      String string0 = CodeGenerator.identifierEscape("com.google.javascript.jscomp.GlobalNamespace$Name");
      assertEquals("com.google.javascript.jscomp.GlobalNamespace$Name", string0);
  }
}
