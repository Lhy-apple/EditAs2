/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:04:01 GMT 2023
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
      String string0 = entities0.entityName(627);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("&$c;&`J1*");
      assertEquals("&$c;&`J1*", string0);
      
      entities0.XML.escape("euml");
      String string1 = entities0.escape("&$c;&`J1*");
      assertEquals("&amp;$c;&amp;`J1*", string1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.XML.escape("euml");
      assertEquals("euml", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Entities.HashEntityMap entities_HashEntityMap0 = new Entities.HashEntityMap();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Entities.TreeEntityMap entities_TreeEntityMap0 = new Entities.TreeEntityMap();
      entities_TreeEntityMap0.add("@D:'::;g", 60);
      int int0 = entities_TreeEntityMap0.value("@D:'::;g");
      assertEquals(60, int0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("&amp;f;8.&amp;`QkJ1*h");
      assertEquals("&f;8.&`QkJ1*h", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Entities.TreeEntityMap entities_TreeEntityMap0 = new Entities.TreeEntityMap();
      int int0 = entities_TreeEntityMap0.value("auml");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Entities entities0 = Entities.XML;
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap();
      Entities.fillWithHtml40Entities(entities0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap(69);
      entities_ArrayEntityMap0.add("", 69);
      String string0 = entities_ArrayEntityMap0.name(69);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap();
      entities_ArrayEntityMap0.add("ivxXXT[hT9c-s Jz", (-2466));
      int int0 = entities_ArrayEntityMap0.value("ivxXXT[hT9c-s Jz");
      assertEquals((-2466), int0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(1369);
      entities_BinaryEntityMap0.add("L7", 55);
      entities_BinaryEntityMap0.add("L7", (-3566));
      entities_BinaryEntityMap0.add("org.apache.commons.lang.IntHashMap$Entry", 55);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("Illegal Capacity: ");
      assertEquals("Illegal Capacity: ", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      StringWriter stringWriter0 = new StringWriter();
      entities0.XML.unescape((Writer) stringWriter0, "&$cRF;n`J1*");
      assertEquals("&$cRF;n`J1*", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Entities entities0 = new Entities();
      // Undeclared exception!
      try { 
        entities0.unescape((Writer) null, "");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.lang.Entities", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("&&c;&`J1*");
      assertEquals("&&c;&`J1*", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.unescape("&;&`kJ1*h");
      assertEquals("&;&`kJ1*h", string0);
  }
}
