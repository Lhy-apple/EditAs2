/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:44:36 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedHashSet;
import java.util.NoSuchElementException;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      document0.appendText("]W?lxF@m3E\"o/oA");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      Element element1 = element0.val("<*>\n<*>\n</*>\n</*>");
      assertFalse(element1.isBlock());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("P0r~#JG");
      Elements elements0 = document0.getElementsByAttributeValueEnding("#comment", "4{p!7Oy+x.,mBx7");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("%>c?>2S!yM");
      Elements elements0 = document0.getElementsByClass("%>c?>2S!yM");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("5E");
      // Undeclared exception!
      try { 
        document0.child(813);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 813, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("Deylta");
      Elements elements0 = document0.getElementsByAttributeValueContaining("Deylta", "Deylta");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      Elements elements0 = document0.getElementsByAttributeValueNot("]W?lxF@m3E\"o/oA", "]W?lxF@m3E\"o/oA");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document(" hqX&\"OW*");
      Elements elements0 = document0.getElementsByAttribute(" hqX&\"OW*");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document(" ");
      // Undeclared exception!
      try { 
        document0.html(" ");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("Could not parse attribute query ");
      Elements elements0 = document0.getElementsByAttributeValue("Could not parse attribute query ", "Could not parse attribute query ");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mB7");
      // Undeclared exception!
      try { 
        document0.siblingElements();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      Elements elements0 = document0.getElementsByAttributeValueStarting("]W?lxF@m3E\"o/oA", "]W?lxF@m3E\"o/oA");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tag tag0 = Tag.valueOf("wJ{O%(aN^");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "qX", attributes0);
      String string0 = element0.nodeName();
      assertEquals("wj{o%(an^", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("3bE[JB SgFJR]=");
      Element element0 = document0.removeClass("3bE[JB SgFJR]=");
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      // Undeclared exception!
      try { 
        document0.normalise();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("<}");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      Elements elements0 = document0.getElementsByIndexLessThan(4);
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("* ");
      Elements elements0 = document0.getElementsByIndexGreaterThan(1938);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("5E");
      Elements elements0 = document0.getElementsByIndexEquals((-1066));
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      Element element0 = document0.prependElement("4{p!7Oy+x.,mBx7");
      Elements elements0 = element0.parents();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      Element element1 = element0.prependElement("* ");
      Elements elements0 = element1.parents();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("5*^aj?^o");
      Element element0 = document0.appendText("5*^aj?^o");
      Element element1 = document0.appendChild(element0);
      Element element2 = element1.previousElementSibling();
      assertNull(element2);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      Element element0 = document0.prependElement("4{p!7Oy+x.,mBx7");
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("5*^aj?^o");
      document0.prependElement("W;85V#P,YK<jI0F");
      Element element0 = document0.appendChild(document0);
      Element element1 = element0.previousElementSibling();
      assertNotNull(element1);
      assertEquals("w;85v#p,yk<ji0f", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      Element element0 = document0.prependElement("4{p!7Oy+x.,mBx7");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("P0r~#JG");
      document0.prependElement("P0r~#JG");
      document0.prependChild(document0);
      Element element0 = document0.firstElementSibling();
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      Element element0 = document0.prependElement("4{p!7Oy+x.,mBx7");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("$z6jr~#,?");
      Element element0 = document0.prependElement("$z6jr~#,?");
      Element element1 = element0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Tag tag0 = Tag.valueOf("*k");
      Element element0 = new Element(tag0, "*k");
      element0.appendChild(element0);
      Element element1 = element0.prependElement("*k");
      Element element2 = element1.lastElementSibling();
      assertSame(element2, element0);
      assertNotNull(element2);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("middot");
      Element element0 = document0.getElementById("middot");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("<}");
      document0.prependElement("<}");
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("fL*AM9?");
      Element element0 = document0.prependElement("textarea");
      element0.prependText("textarea");
      String string0 = element0.val();
      assertEquals("textarea", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      Element element1 = element0.prependText(";/@j!T!hDz%d'&i");
      element1.prependText("* ");
      String string0 = element1.text();
      assertEquals("* ;/@j!T!hDz%d'&i", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("<}");
      document0.prependElement("<}");
      Element element0 = document0.prependText("<}");
      String string0 = element0.text();
      assertEquals("<}", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("5E");
      Element element0 = document0.prependElement("5E");
      boolean boolean0 = element0.preserveWhitespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Tag tag0 = Tag.valueOf("P6Jr~#JG");
      Element element0 = new Element(tag0, "Gk4U:qx@zY");
      Element element1 = element0.prependText("");
      boolean boolean0 = element1.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      element0.prependElement("* ");
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("5E");
      Comment comment0 = new Comment("e0;>q+P9|=1^8GY'<", "tz:+t^08D&7");
      document0.appendChild(comment0);
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("P");
      Element element0 = document0.prependElement("P");
      element0.appendText("P");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      DataNode dataNode0 = new DataNode(">&YC)M.", "");
      document0.prependChild(dataNode0);
      String string0 = document0.data();
      assertEquals(">&YC)M.", string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      Element element0 = document0.prependElement("class");
      assertEquals("class", element0.nodeName());
      
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("<@Yxywg,MXY<");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Element element0 = document0.classNames((Set<String>) linkedHashSet0);
      boolean boolean0 = element0.hasClass("<@Yxywg,MXY<");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("<}");
      document0.classNames();
      Element element0 = document0.addClass("");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      boolean boolean0 = document0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("P6Jr~#JG");
      Element element0 = document0.toggleClass("P6Jr~#JG");
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4{p!7Oy+x.,mBx7");
      Element element0 = new Element(tag0, "J");
      Element element1 = element0.toggleClass("");
      assertSame(element0, element1);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Element element0 = new Element(tag0, "org.jsoup.nodes.Element");
      Element element1 = element0.val("I;2NN;?@Z~/KWx<wI7_");
      assertEquals("org.jsoup.nodes.Element", element1.baseUri());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("TITLE");
      document0.prependElement("TITLE");
      String string0 = document0.toString();
      assertEquals("<title></title>", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("<}");
      document0.prependElement("<}");
      document0.prependText("<}");
      String string0 = document0.toString();
      assertEquals("&lt;}<<}>\n</<}>", string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      Element element1 = element0.prependElement("* ");
      String string0 = element0.toString();
      assertFalse(element0.equals((Object)element1));
      assertEquals("<*>\n<*>\n</*>\n</*>", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("PARAM");
      Element element0 = document0.prependElement("PARAM");
      String string0 = element0.toString();
      assertEquals("\n<param />", string0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("5E");
      boolean boolean0 = document0.equals("5E");
      assertFalse(boolean0);
  }
}