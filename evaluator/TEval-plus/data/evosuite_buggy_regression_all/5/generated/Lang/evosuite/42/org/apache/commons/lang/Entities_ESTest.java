/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:25:15 GMT 2023
 */

package org.apache.commons.lang;

import org.junit.Test;
import static org.junit.Assert.*;
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
      String string0 = entities0.unescape("!:7bE&;WM");
      assertEquals("!:7bE&;WM", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("qdCh P&quot;R&amp;&lt;}");
      assertEquals("qdCh P\"R&<}", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      Entities.fillWithHtml40Entities(entities0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Entities.HashEntityMap entities_HashEntityMap0 = new Entities.HashEntityMap();
      entities_HashEntityMap0.add("Vm76`", (-2630));
      int int0 = entities_HashEntityMap0.value("Vm76`");
      assertEquals((-2630), int0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Entities.TreeEntityMap entities_TreeEntityMap0 = new Entities.TreeEntityMap();
      String string0 = entities_TreeEntityMap0.name((-52));
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Entities.HashEntityMap entities_HashEntityMap0 = new Entities.HashEntityMap();
      int int0 = entities_HashEntityMap0.value("Vm76`");
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      String string0 = entities0.entityName(666);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Entities entities0 = Entities.XML;
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap();
      String string0 = entities0.escape("z<n@&(}N;ao");
      assertEquals("z&lt;n@&amp;(}N;ao", string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Entities entities0 = Entities.XML;
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap();
      entities0.map = (Entities.EntityMap) entities_ArrayEntityMap0;
      Entities.fillWithHtml40Entities(entities0);
      String string0 = entities0.escape("z<n@&(}N;ao");
      assertEquals("z&lt;n@&amp;(}N;ao", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Entities entities0 = Entities.XML;
      Entities.ArrayEntityMap entities_ArrayEntityMap0 = new Entities.ArrayEntityMap();
      entities0.map = (Entities.EntityMap) entities_ArrayEntityMap0;
      Entities.fillWithHtml40Entities(entities0);
      String string0 = entities0.unescape("z<n@&(}N;ao");
      assertEquals("z<n@&(}N;ao", string0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(111);
      entities_BinaryEntityMap0.add("u@_", 111);
      int int0 = entities_BinaryEntityMap0.value("u@_");
      assertEquals(111, int0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(111);
      entities_BinaryEntityMap0.add("u@_", 111);
      String string0 = entities_BinaryEntityMap0.name(Integer.MAX_VALUE);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap();
      entities_BinaryEntityMap0.add((String) null, (-1));
      entities_BinaryEntityMap0.add((String) null, 1162);
      entities_BinaryEntityMap0.add((String) null, 1162);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Entities.BinaryEntityMap entities_BinaryEntityMap0 = new Entities.BinaryEntityMap(128);
      entities_BinaryEntityMap0.add("bu", 128);
      String string0 = entities_BinaryEntityMap0.name(128);
      assertEquals("bu", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Entities entities0 = Entities.HTML32;
      String string0 = entities0.unescape("Vm76`");
      assertEquals("Vm76`", string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(1368);
      Entities entities0 = Entities.HTML40;
      entities0.unescape((Writer) stringWriter0, " ih$P\"R&<}");
      assertEquals(" ih$P\"R&<}", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      StringWriter stringWriter0 = new StringWriter(1368);
      Entities entities0 = Entities.HTML40;
      entities0.unescape((Writer) stringWriter0, "t.FrQeT");
      assertEquals("t.FrQeT", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Entities entities0 = Entities.XML;
      String string0 = entities0.unescape("q9J=P,aewPoa=gs&&wZ;");
      assertEquals("q9J=P,aewPoa=gs&&wZ;", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Entities entities0 = new Entities();
      String string0 = entities0.unescape("gq7J{aL=gZ&#$;?");
      assertEquals("gq7J{aL=gZ&#$;?", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Entities entities0 = Entities.HTML40;
      String string0 = entities0.unescape("gq7J{aL=gZ&#;?");
      assertEquals("gq7J{aL=gZ&#;?", string0);
  }
}
