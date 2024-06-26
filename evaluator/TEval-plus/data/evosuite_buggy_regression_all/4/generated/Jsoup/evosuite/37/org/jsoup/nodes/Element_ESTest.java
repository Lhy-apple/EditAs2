/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:53:24 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.nodes.XmlDeclaration;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      Element element0 = formElement0.appendElement("h1");
      element0.prependText("textarea");
      formElement0.getElementsContainingOwnText("ScriptDataDoubleEscapedDash");
      assertEquals(1, formElement0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("4GKR$s|K7'2G");
      Element element0 = document0.appendText("4GKR$s|K7'2G");
      assertEquals("4GKR$s|K7'2G", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("title");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "title", attributes0);
      formElement0.appendElement("title");
      Element element0 = formElement0.prependText("title");
      String string0 = element0.text();
      assertEquals("title", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf("I@$w{V1s");
      Element element0 = new Element(tag0, "I@$w{V1s");
      // Undeclared exception!
      try { 
        element0.child(3149);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 3149, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      Element element0 = formElement0.appendElement("h1");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "textarea");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      element0.outerHtmlHead(stringBuilder0, (-1137), document_OutputSettings0);
      assertEquals("textarea<h1>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("");
      // Undeclared exception!
      try { 
        document0.tagName("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Tag name must not be empty.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("#root");
      // Undeclared exception!
      try { 
        document0.html("#root");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("P|mPN<i+wDjo=8");
      Map<String, String> map0 = document0.dataset();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("Ir$iWp#&:CWGT$");
      Elements elements0 = document0.getElementsByAttributeValue("Ir$iWp#&:CWGT$", "Ir$iWp#&:CWGT$");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("^Ab");
      Elements elements0 = document0.getElementsByAttributeValueStarting("^Ab", "^Ab");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("&WVdQ(6k_E@WOB<");
      // Undeclared exception!
      try { 
        document0.select("&WVdQ(6k_E@WOB<");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query '&WVdQ(6k_E@WOB<': unexpected token at '&WVdQ(6k_E@WOB<'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("C-73U`qqaNz");
      Element element0 = document0.prepend("C-73U`qqaNz");
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("C-73U`qqaNz");
      Elements elements0 = document0.getElementsMatchingOwnText("C-73U`qqaNz");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("qY&^7jjk^");
      // Undeclared exception!
      try { 
        document0.after("qY&^7jjk^");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("~><-dv0");
      Elements elements0 = document0.getElementsByIndexLessThan(1);
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("~><-0");
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
  public void test16()  throws Throwable  {
      Document document0 = new Document("4GKR$s|Ka7'2G");
      Elements elements0 = document0.getElementsByAttributeStarting("4GKR$s|Ka7'2G");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("ma:UE");
      Elements elements0 = document0.getElementsByIndexEquals((-1007));
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("ma:UE");
      // Undeclared exception!
      try { 
        document0.wrap("ma:UE");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("~><-dv0");
      Document document1 = (Document)document0.val("~><-dv0");
      assertEquals("~><-dv0", document1.location());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("^/b");
      Elements elements0 = document0.getElementsByAttributeValueMatching("^/b", "^/b");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("xtarea");
      Elements elements0 = document0.getElementsByAttributeValueEnding("xtarea", "xtarea");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document(",^<CfCXU6^\"HKDn'}aU");
      Elements elements0 = document0.getElementsByClass(",^<CfCXU6^\"HKDn'}aU");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("4GKR$s|Ka7'2G");
      Elements elements0 = document0.getElementsContainingText("4GKR$s|Ka7'2G");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("id");
      Elements elements0 = document0.getElementsByAttributeValueContaining("id", "id");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("~><-dv0");
      Document document1 = document0.clone();
      assertNotSame(document1, document0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("textarea");
      // Undeclared exception!
      try { 
        document0.getElementsByAttributeValueNot((String) null, (String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("\"7`\"@@k}7oi`");
      Elements elements0 = document0.getElementsByAttribute("\"7`\"@@k}7oi`");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document(",/jb,:");
      // Undeclared exception!
      try { 
        document0.before(",/jb,:");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("C-73U`qqaNz");
      Document document1 = (Document)document0.removeClass("C-73U`qqaNz");
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Tag tag0 = Tag.valueOf("I@$w{V1s");
      Element element0 = new Element(tag0, "I@$w{V1s");
      Elements elements0 = element0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("~><-dv0");
      Elements elements0 = document0.getElementsMatchingText("~><-dv0");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("~><-dv0");
      // Undeclared exception!
      try { 
        document0.title("~><-dv0");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("H");
      Elements elements0 = document0.getElementsByIndexGreaterThan(2507);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("4GK$|Kag'2G");
      Node[] nodeArray0 = new Node[4];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      DataNode dataNode0 = DataNode.createFromEncoded(")2*o~?H.%Dzf'cZnsE ", "4GK$|Kag'2G");
      nodeArray0[2] = (Node) dataNode0;
      nodeArray0[3] = (Node) document0;
      document0.addChildren(nodeArray0);
      document0.append("4GK$|Kag'2G");
      assertEquals(3, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Nc>#^");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Nc>#^", attributes0);
      Element element0 = formElement0.appendElement("Nc>#^");
      // Undeclared exception!
      try { 
        element0.append("Nc>#^");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Hv");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Hv", attributes0);
      Element element0 = formElement0.appendElement("Hv");
      element0.before((Node) formElement0);
      Element element1 = formElement0.prependText("Hv");
      Element element2 = element1.nextElementSibling();
      assertEquals("Hv", element2.baseUri());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Tag tag0 = Tag.valueOf("title");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "title", attributes0);
      formElement0.appendElement("title");
      List<TextNode> list0 = formElement0.textNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Tag tag0 = Tag.valueOf("title");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "title", attributes0);
      Element element0 = formElement0.prependText("title");
      List<TextNode> list0 = element0.textNodes();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("4GK$|Kag'2G");
      Node[] nodeArray0 = new Node[4];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      DataNode dataNode0 = DataNode.createFromEncoded(")2*o~?H.%Dzf'cZnsE ", "4GK$|Kag'2G");
      nodeArray0[2] = (Node) dataNode0;
      nodeArray0[3] = (Node) document0;
      document0.addChildren(nodeArray0);
      List<DataNode> list0 = document0.dataNodes();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("!+;gUN1E");
      LinkedList<FormElement> linkedList0 = new LinkedList<FormElement>();
      // Undeclared exception!
      try { 
        document0.insertChildren((-1074265344), linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("&WVdQ(6k_E@WOB<");
      LinkedList<XmlDeclaration> linkedList0 = new LinkedList<XmlDeclaration>();
      // Undeclared exception!
      try { 
        document0.insertChildren(913, linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("5@$w}v1;");
      Elements elements0 = document0.siblingElements();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4GR$|K7'2G");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "4GR$|K7'2G", attributes0);
      Element element0 = formElement0.appendElement("4GR$|K7'2G");
      element0.after((Node) formElement0);
      Elements elements0 = element0.siblingElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      Element element0 = formElement0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Tag tag0 = Tag.valueOf("!+;gUN1E");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "!+;gUN1E", attributes0);
      Element element0 = formElement0.appendElement("!+;gUN1E");
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4GKR>$s|Ka7'2G");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "4GKR>$s|Ka7'2G", attributes0);
      Element element0 = formElement0.appendElement("-';ByVOkV8=v");
      Element element1 = element0.previousElementSibling();
      assertNull(element1);
      assertEquals("-';byvokv8=v", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("6D%2BRK");
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4GKR>$s|Ka7'2G");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "4GKR>$s|Ka7'2G", attributes0);
      Element element0 = formElement0.appendElement("-';ByVOkV8=v");
      element0.before((Node) formElement0);
      Element element1 = element0.previousElementSibling();
      assertEquals("-';byvokv8=v", element0.tagName());
      assertNotNull(element1);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("4gkr$s|ka7'2g");
      document0.appendChild(document0);
      Element element0 = document0.firstElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4FKR>$s|Ka7'2G");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "4FKR>$s|Ka7'2G", attributes0);
      Node[] nodeArray0 = new Node[6];
      DataNode dataNode0 = new DataNode("4gk$|kag'2g", "/+*(");
      nodeArray0[0] = (Node) dataNode0;
      nodeArray0[1] = (Node) formElement0;
      nodeArray0[2] = (Node) formElement0;
      nodeArray0[3] = (Node) formElement0;
      nodeArray0[4] = (Node) formElement0;
      nodeArray0[5] = (Node) formElement0;
      formElement0.addChildren(nodeArray0);
      Element element0 = formElement0.appendElement("4FKR>$s|Ka7'2G");
      Element element1 = element0.firstElementSibling();
      assertSame(element1, formElement0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Tag tag0 = Tag.valueOf("!+gUN1E");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "!+gUN1E", attributes0);
      Element element0 = formElement0.appendElement("!+gUN1E");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "textarea", attributes0);
      Element element0 = formElement0.appendElement("textarea");
      Element element1 = element0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4GKR>$s|Ka7'2G");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "beforeattributevalue", attributes0);
      Node[] nodeArray0 = new Node[8];
      nodeArray0[0] = (Node) formElement0;
      nodeArray0[1] = (Node) formElement0;
      nodeArray0[2] = (Node) formElement0;
      nodeArray0[3] = (Node) formElement0;
      Comment comment0 = new Comment("4GKR>$s|Ka7'2G", "n#^");
      nodeArray0[4] = (Node) comment0;
      nodeArray0[5] = (Node) formElement0;
      nodeArray0[6] = (Node) formElement0;
      nodeArray0[7] = (Node) formElement0;
      formElement0.addChildren(nodeArray0);
      Element element0 = formElement0.appendElement("`GlIr|");
      Element element1 = element0.lastElementSibling();
      assertNotNull(element1);
      assertEquals("`glir|", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "textarea", attributes0);
      Element element0 = formElement0.appendElement("textarea");
      formElement0.val("H[*=CsrFQ");
      // Undeclared exception!
      try { 
        element0.nextElementSibling();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      Element element0 = formElement0.getElementById("ScriptDataDoubleEscapedDash");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = new Document("zib");
      Comment comment0 = new Comment("zib", "xs7w5lTs~E(g]X");
      document0.prependChild(comment0);
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Tag tag0 = Tag.valueOf("p");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "p", attributes0);
      Element element0 = formElement0.appendElement("br");
      assertEquals("br", element0.nodeName());
      
      Element element1 = formElement0.prependText("br");
      String string0 = element1.text();
      assertEquals("br", string0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Hv");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Hv", attributes0);
      formElement0.appendElement("Hv");
      Element element0 = formElement0.prependText("Hv");
      String string0 = element0.text();
      assertEquals("Hv", string0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Tag tag0 = Tag.valueOf("h2");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h2", attributes0);
      Element element0 = formElement0.appendElement("h2");
      Element element1 = formElement0.prependText("h2");
      element0.appendChild(formElement0);
      element1.text();
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "textarea", attributes0);
      LinkedList<Comment> linkedList0 = new LinkedList<Comment>();
      Comment comment0 = new Comment("numeric reference with no numerals", "textarea");
      linkedList0.add(comment0);
      formElement0.insertChildren((-1), linkedList0);
      Elements elements0 = formElement0.getElementsContainingOwnText("ScriptDataDoubleEscapedDash");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "textarea", attributes0);
      formElement0.appendElement("br");
      Elements elements0 = formElement0.getElementsContainingOwnText("'*/!O&n/");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      boolean boolean0 = Element.preserveWhitespace((Node) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      DataNode dataNode0 = DataNode.createFromEncoded("}:;}y3x.@suX&l", "@udswvww[5pl");
      boolean boolean0 = Element.preserveWhitespace(dataNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      formElement0.appendElement("h1");
      boolean boolean0 = formElement0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Tag tag0 = Tag.valueOf("org.jsoup.nodes.Element$1");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "", attributes0);
      formElement0.prependText("");
      boolean boolean0 = formElement0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "textarea", attributes0);
      LinkedList<Comment> linkedList0 = new LinkedList<Comment>();
      Comment comment0 = new Comment("numeric reference with no numerals", "textarea");
      linkedList0.add(comment0);
      formElement0.insertChildren((-1), linkedList0);
      boolean boolean0 = formElement0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      Element element0 = formElement0.appendElement("h1");
      element0.prependText("textarea");
      boolean boolean0 = formElement0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4FKR>$s|Ka7'2G");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "4FKR>$s|Ka7'2G", attributes0);
      Node[] nodeArray0 = new Node[6];
      DataNode dataNode0 = new DataNode("4gk$|kag'2g", "/+*(");
      nodeArray0[0] = (Node) dataNode0;
      nodeArray0[1] = (Node) formElement0;
      nodeArray0[2] = (Node) formElement0;
      nodeArray0[3] = (Node) formElement0;
      nodeArray0[4] = (Node) formElement0;
      nodeArray0[5] = (Node) formElement0;
      formElement0.addChildren(nodeArray0);
      // Undeclared exception!
      formElement0.data();
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Document document0 = new Document("4gkr$s|ka7'2g");
      document0.append("4gkr$s|ka7'2g");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Document document0 = new Document("OWo'{r?dxPVv");
      document0.addClass("OWo'{r?dxPVv");
      boolean boolean0 = document0.hasClass("OWo'{r?dxPVv");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = new Document("P|mPN<i+wDj=%o@8");
      Element element0 = document0.toggleClass("P|mPN<i+wDj=%o@8");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Tag tag0 = Tag.valueOf("4GKR>>s|Ka7^G");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "4GKR>>s|Ka7^G", attributes0);
      Element element0 = formElement0.toggleClass("");
      assertEquals("4GKR>>s|Ka7^G", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Document document0 = new Document("~><-dv0");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "textarea", attributes0);
      String string0 = formElement0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Document document0 = new Document("5@$w}v1;");
      StringBuilder stringBuilder0 = new StringBuilder(" />");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlHead(stringBuilder0, 1030, document_OutputSettings1);
      assertEquals(" /><#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Document document0 = new Document("u?");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "u?");
      document0.outerHtml(stringBuilder0);
      assertEquals("u?\n<#root></#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Tag tag0 = Tag.valueOf("p");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "p", attributes0);
      StringBuilder stringBuilder0 = new StringBuilder("p");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      formElement0.outerHtmlHead(stringBuilder0, 80, document_OutputSettings0);
      assertEquals("p<p>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Tag tag0 = Tag.valueOf("!;gN1");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "!;gN1", attributes0);
      formElement0.appendElement("br");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "br");
      // Undeclared exception!
      try { 
        stringBuilder0.insert(1013, (Object) formElement0);
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "textarea");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        formElement0.outerHtmlHead(stringBuilder0, (-1137), document_OutputSettings0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // width must be > 0
         //
         verifyException("org.jsoup.helper.StringUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Document document0 = new Document("4GR$|K7'2G");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "4GR$|K7'2G");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlTail(stringBuilder0, 46, document_OutputSettings0);
      assertEquals("4GR$|K7'2G</#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Tag tag0 = Tag.valueOf("p");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "p", attributes0);
      formElement0.appendElement("\"MT");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      // Undeclared exception!
      try { 
        formElement0.outerHtmlTail((StringBuilder) null, (-191), document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Tag tag0 = Tag.valueOf("p");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "p", attributes0);
      formElement0.appendElement("\"MT");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        formElement0.outerHtmlTail((StringBuilder) null, (-191), document_OutputSettings1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Tag tag0 = Tag.valueOf("p");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "p", attributes0);
      Element element0 = formElement0.appendElement("p");
      element0.before((Node) formElement0);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        formElement0.outerHtmlTail((StringBuilder) null, 61, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Tag tag0 = Tag.valueOf("p");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "p", attributes0);
      Element element0 = formElement0.appendElement("\"MT");
      Element element1 = element0.prependText("\"MT");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      element1.unwrap();
      // Undeclared exception!
      try { 
        formElement0.outerHtmlTail((StringBuilder) null, (-191), document_OutputSettings1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      Document document0 = new Document("qi`'oNh");
      Element element0 = document0.append("qi`'oNh");
      String string0 = element0.toString();
      assertEquals("qi`'oNh", string0);
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "h1", attributes0);
      formElement0.hashCode();
  }
}
