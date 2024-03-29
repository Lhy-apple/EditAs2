/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:24:13 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import java.io.Writer;
import org.apache.commons.lang.Entities;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Entities_ESTest extends Entities_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Entities entities0 = new Entities();
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(2946);
      entities0.map = (Entities.EntityMap) entities_BinaryEntityMap0;
      Entities.fillWithHtml40Entities(entities0);
      String string0 = entities0.escape("!C83nn&(,ml9p=w'l");
      assertEquals("!C83nn&amp;(,ml9p=w'l", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.escape("~R&#wYC;");
      assertEquals("~R&amp;#wYC;", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      Entities.fillWithHtml40Entities(entities0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap();
      entities_BinaryEntityMap0.add("AZreHPB[3", 97);
      int int0 = entities_BinaryEntityMap0.value("AZreHPB[3");
      assertEquals(97, int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Entities.TreeEntityMap entities_TreeEntityMap0 = new Entities.TreeEntityMap();
      // Undeclared exception!
      try { 
        entities_TreeEntityMap0.add((String) null, (-961));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.TreeMap", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Entities.HashEntityMap entities_HashEntityMap0 = new Entities.HashEntityMap();
      entities_HashEntityMap0.add("supe", 55);
      int int0 = entities_HashEntityMap0.value("supe");
      assertEquals(55, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.unescape("]*&!;");
      assertEquals("]*&!;", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Entities.HashEntityMap entities_HashEntityMap0 = new Entities.HashEntityMap();
      int int0 = entities_HashEntityMap0.value("supe");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.XML.entityName(1865);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(1546);
      int[] intArray0 = new int[6];
      intArray0[0] = 1546;
      entities_BinaryEntityMap0.add("-p\"E#?.s~P}wY]QP{", 2);
      entities_BinaryEntityMap0.values = intArray0;
      Entities entities0 = Entities.HTML32;
      entities0.HTML32.map = (Entities.EntityMap) entities_BinaryEntityMap0;
      entities0.HTML32.addEntity("-p\"E#?.s~P}wY]QP{", 2);
      entities_BinaryEntityMap0.add("Xxa", 1546);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Entities entities0 = Entities.XML;
      StringWriter stringWriter0 = new StringWriter();
      entities0.escape((Writer) stringWriter0, "&&5V;*;");
      assertEquals("&amp;&amp;5V;*;", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Entities entities0 = new Entities();
      String string0 = entities0.unescape("org.apache.commons.lang.IntHashMap");
      assertEquals("org.apache.commons.lang.IntHashMap", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("&apos;el&quot;I1");
      assertEquals("'el\"I1", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("3~s$a@OxOTB&8v$:P-");
      assertEquals("3~s$a@OxOTB&8v$:P-", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.unescape("&&;");
      assertEquals("&&;", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("&#rC.;");
      assertEquals("&#rC.;", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("]&#;");
      assertEquals("]&#;", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Entities entities0 = new Entities();
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "-p\"E#?.s~P}wY]QP{");
      assertEquals("-p\"E#?.s~P}wY]QP{", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Entities entities0 = Entities.XML;
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "&&5");
      assertEquals("&&5", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      Entities entities0 = Entities.HTML32;
      entities0.unescape((Writer) stringWriter0, "&&;");
      assertEquals("&&;", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      Entities entities0 = Entities.HTML32;
      entities0.unescape((Writer) stringWriter0, "&amp;&amp; !;;");
      assertEquals("&& !;;", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "&#YC;");
      assertEquals("&#YC;", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "]&#;");
      assertEquals("]&#;", stringWriter0.toString());
  }
}
