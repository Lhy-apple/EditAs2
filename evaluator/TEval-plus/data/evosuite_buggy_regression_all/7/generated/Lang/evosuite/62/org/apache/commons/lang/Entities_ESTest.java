/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:46:33 GMT 2023
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
      Entities entities0 = Entities.HTML32;
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "}&gt;A0&amp;005]cU~uf");
      assertEquals("}>A0&005]cU~uf", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      Entities.fillWithHtml40Entities(entities0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap();
      entities_ArrayEntityMap0.add("\"CTPHi={o=;g)", 235);
      int int0 = entities_ArrayEntityMap0.value("\"CTPHi={o=;g)");
      assertEquals(235, int0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(43);
      entities_BinaryEntityMap0.size = 43;
      entities_BinaryEntityMap0.add("Illegal Load: ", 43);
      entities_BinaryEntityMap0.add("Illegal Load: ", 43);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap();
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Entities.HashEntityMap entities_HashEntityMap0 = new Entities.HashEntityMap();
      entities_HashEntityMap0.add((String) null, 20);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Entities.TreeEntityMap entities_TreeEntityMap0 = new Entities.TreeEntityMap();
      entities_TreeEntityMap0.add("Z(aOBk", (-3175));
      int int0 = entities_TreeEntityMap0.value("Z(aOBk");
      assertEquals((-3175), int0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Entities entities0 = Entities.XML;
      Entities.TreeEntityMap entities_TreeEntityMap0 = new Entities.TreeEntityMap();
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "r(gf&$/XoR20;Nf");
      assertEquals("r(gf&$/XoR20;Nf", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.HTML32.entityName(1343);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap();
      entities0.map = (Entities.EntityMap) entities_ArrayEntityMap0;
      Entities.fillWithHtml40Entities(entities0);
      String string0 = entities0.escape("!loq(XJ0%sE(\"r=");
      assertEquals("!loq(XJ0%sE(&quot;r=", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap(2919);
      entities_ArrayEntityMap0.add("*L Pj#f&", (-1087));
      int int0 = entities_ArrayEntityMap0.value("~8ol8+Y{da]WJ=!m0");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(43);
      entities_BinaryEntityMap0.add("Illegal Load: ", 43);
      // Undeclared exception!
      try { 
        entities_BinaryEntityMap0.add("Illegal Load: ", 43);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      StringWriter stringWriter0 = new StringWriter();
      entities0.escape((Writer) stringWriter0, "\"CTPHi={o=;g)");
      assertEquals("&quot;CTPHi={o=;g)", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      String string0 = entities0.unescape(">&5zv,(b");
      assertEquals(">&5zv,(b", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.unescape("9#TPH={T%^oe=z;g)");
      assertEquals("9#TPH={T%^oe=z;g)", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.unescape("2p5r?&;!f");
      assertEquals("2p5r?&;!f", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("}&gt;A0&amp;005]cU~uf");
      assertEquals("}>A0&005]cU~uf", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter();
      Entities entities0 = Entities.XML;
      entities0.unescape((Writer) stringWriter0, "\"CTPHi={o=;g)");
      assertEquals("\"CTPHi={o=;g)", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "}>A0&005]cU~uf");
      assertEquals("}>A0&005]cU~uf", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Entities entities0 = new Entities();
      StringWriter stringWriter0 = new StringWriter();
      entities0.unescape((Writer) stringWriter0, "nY2&;");
      assertEquals("nY2&;", stringWriter0.toString());
  }
}
