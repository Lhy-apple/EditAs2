/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:22:23 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      document0.append("uz");
      document0.appendChild(document0);
      document0.getElementsMatchingText("Pattern syntax error: ");
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("odd");
      // Undeclared exception!
      try { 
        document0.child((-841));
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("p");
      document0.appendElement("p");
      document0.appendChild(document0);
      document0.html();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("~X");
      Element element0 = document0.prependText((String) null);
      List<TextNode> list0 = element0.textNodes();
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("or5.jsoup.elect.Evaluator$Conta\nOwTexz");
      Document document1 = (Document)document0.tagName("or5.jsoup.elect.Evaluator$Conta\nOwTexz");
      assertEquals("or5.jsoup.elect.Evaluator$Conta\nOwTexz", document1.location());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("~X");
      Map<String, String> map0 = document0.dataset();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("org.jsoup.parser.HtmlTreeBuilderState$24");
      document0.addClass("org.jsoup.parser.HtmlTreeBuilderState$24");
      Element element0 = document0.toggleClass("org.jsoup.parser.HtmlTreeBuilderState$24");
      Document document1 = (Document)element0.toggleClass("org.jsoup.parser.HtmlTreeBuilderState$24");
      assertFalse(document1.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document(":,=rG2e1");
      // Undeclared exception!
      try { 
        document0.before((Node) document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("Ez");
      // Undeclared exception!
      try { 
        document0.html("Ez");
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
      Document document0 = new Document("E9z");
      Elements elements0 = document0.getElementsByAttributeValue("E9z", "E9z");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("z");
      Elements elements0 = document0.getElementsByAttributeValueStarting("z", "p~tx2");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      Element element0 = document0.createElement("h6");
      Element element1 = element0.appendElement("h6");
      String string0 = element1.cssSelector();
      assertEquals(0, element1.siblingIndex());
      assertEquals("h6 > h6", string0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("type");
      Element element0 = document0.prepend("type");
      Document document1 = new Document("type");
      document1.appendChild(element0);
      String string0 = document1.html();
      assertEquals("<#root>\n type\n</#root>", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("zMqKNo?,<~3WH");
      // Undeclared exception!
      try { 
        document0.after("zMqKNo?,<~3WH");
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
      Document document0 = new Document("type");
      Elements elements0 = document0.getElementsByIndexLessThan(50);
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$ContainsOwnText");
      Elements elements0 = document0.getElementsByAttributeStarting("org.jsoup.select.Evaluator$ContainsOwnText");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("~X");
      Elements elements0 = document0.getElementsByIndexEquals(3154);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("org.jsoup.selecz.Evaluator$ContainsOwnText");
      // Undeclared exception!
      try { 
        document0.wrap("org.jsoup.selecz.Evaluator$ContainsOwnText");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("java.lang.string@0000000008");
      Elements elements0 = document0.getElementsByAttributeValueMatching("$i:r", "java.lang.string@0000000008");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("3!IDqWh:dX");
      // Undeclared exception!
      try { 
        document0.getElementsByAttributeValueEnding("", "");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("~X");
      Elements elements0 = document0.getElementsByClass("~X");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("type");
      Elements elements0 = document0.getElementsContainingText("CqPJwb]egF]{");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("org.jsoup.nodes.Element$1");
      Elements elements0 = document0.getElementsByAttributeValueContaining("org.jsoup.nodes.Element$1", "org.jsoup.nodes.Element$1");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$ContainsOwnText");
      // Undeclared exception!
      try { 
        document0.after((Node) document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("type");
      Elements elements0 = document0.getElementsByAttributeValueNot("type", "type");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("type");
      String string0 = document0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("~X");
      Elements elements0 = document0.getElementsByAttribute("~X");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("|w-s#L$5?");
      // Undeclared exception!
      try { 
        document0.before("|w-s#L$5?");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.removeClass("");
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("VI(4[2rv14U3io");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("");
      // Undeclared exception!
      try { 
        document0.getElementsByTag("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("type");
      Element element0 = document0.createElement("u");
      element0.reparentChild(document0);
      Element element1 = document0.prepend("type");
      assertEquals("#root", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("typS");
      Element element0 = document0.appendChild(document0);
      document0.append("09C(!");
      element0.firstElementSibling();
      assertEquals(2, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("~X");
      document0.appendChild(document0);
      List<TextNode> list0 = document0.textNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("Vk&7");
      DataNode dataNode0 = DataNode.createFromEncoded("org.jsoup.select.Evaluator$ContainsOwnText", "Vk&7");
      document0.appendChild(dataNode0);
      List<DataNode> list0 = document0.dataNodes();
      assertTrue(list0.contains(dataNode0));
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$ContainsOwnText");
      document0.append("org.jsoup.select.Evaluator$ContainsOwnText");
      List<DataNode> list0 = document0.dataNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("type");
      LinkedHashSet<FormElement> linkedHashSet0 = new LinkedHashSet<FormElement>();
      // Undeclared exception!
      try { 
        document0.insertChildren(9863, linkedHashSet0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("yp");
      LinkedHashSet<DocumentType> linkedHashSet0 = new LinkedHashSet<DocumentType>();
      Element element0 = document0.insertChildren((-1), linkedHashSet0);
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$ContainsOwnText");
      LinkedList<DocumentType> linkedList0 = new LinkedList<DocumentType>();
      // Undeclared exception!
      try { 
        document0.insertChildren((-174), linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("typ");
      document0.toggleClass("typ");
      String string0 = document0.cssSelector();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("type");
      Element element0 = document0.appendChild(document0);
      String string0 = element0.cssSelector();
      assertEquals("#root", string0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.CombiningEvaluator$Or");
      Elements elements0 = document0.siblingElements();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("qtype");
      document0.prependElement("qtype");
      document0.appendChild(document0);
      document0.siblingElements();
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("typ@");
      Element element0 = document0.appendChild(document0);
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("E9z");
      Element element0 = document0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("typ@");
      Element element0 = document0.appendChild(document0);
      element0.appendElement("textarea");
      element0.nextElementSibling();
      assertEquals(2, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("type");
      document0.appendChild(document0);
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$ContainsOwnText");
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("type");
      Element element0 = document0.appendChild(document0);
      element0.prependElement("CqPJwb]egF]{");
      document0.previousElementSibling();
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("typS");
      Tag tag0 = Tag.valueOf("typS");
      Element element0 = new Element(tag0, "typS");
      document0.prependChild(element0);
      Element element1 = document0.appendChild(document0);
      element1.firstElementSibling();
      assertEquals(1, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("type");
      Element element0 = document0.appendChild(document0);
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("type");
      document0.appendChild(document0);
      Element element0 = document0.lastElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("type");
      Element element0 = document0.appendChild(document0);
      element0.prependElement("var");
      element0.lastElementSibling();
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("type");
      document0.reparentChild(document0);
      // Undeclared exception!
      try { 
        document0.nextElementSibling();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("yp");
      Element element0 = document0.getElementById("yp");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = new Document("ypzju2b");
      DataNode dataNode0 = DataNode.createFromEncoded("ypzju2b", "ypzju2b");
      document0.appendChild(dataNode0);
      Elements elements0 = document0.getElementsMatchingText("java.lang.string@0000000028");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = new Document("p");
      Element element0 = document0.createElement("p");
      document0.append("p");
      document0.appendChild(element0);
      Elements elements0 = document0.getElementsMatchingText("p");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      Element element0 = document0.createElement("br");
      document0.append("uz");
      document0.appendChild(element0);
      Elements elements0 = document0.getElementsMatchingText("Pattern syntax error: ");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = new Document("br");
      document0.appendChild(document0);
      // Undeclared exception!
      document0.getElementsMatchingOwnText("br");
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("ype");
      Element element0 = document0.append("ype");
      Elements elements0 = element0.getElementsContainingOwnText("ype");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("ype");
      DataNode dataNode0 = new DataNode("ype", "ype");
      document0.appendChild(dataNode0);
      Elements elements0 = document0.getElementsContainingOwnText("ype");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.createElement("br");
      document0.appendChild(element0);
      Elements elements0 = document0.getElementsMatchingOwnText("br");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      boolean boolean0 = Element.preserveWhitespace((Node) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      TextNode textNode0 = new TextNode("#", "#");
      boolean boolean0 = Element.preserveWhitespace(textNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "type", attributes0);
      boolean boolean0 = Element.preserveWhitespace(formElement0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.CombiningEvaluator$Or");
      document0.appendElement("org.jsoup.select.CombiningEvaluator$Or");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.CombiningEvaluator$Or");
      Element element0 = document0.appendElement("org.jsoup.select.CombiningEvaluator$Or");
      element0.text("8fQU;`w");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Document document0 = new Document("~X");
      document0.prependText((String) null);
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Document document0 = new Document("type");
      DataNode dataNode0 = new DataNode("type", "TjIOL\":;}jN");
      document0.appendChild(dataNode0);
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Document document0 = new Document("Qc6XyB2R!A%9:&bI");
      DataNode dataNode0 = new DataNode("Qc6XyB2R!A%9:&bI", "Qc6XyB2R!A%9:&bI");
      document0.appendChild(dataNode0);
      String string0 = document0.data();
      assertEquals("Qc6XyB2R!A%9:&bI", string0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Document document0 = new Document("type");
      document0.appendChild(document0);
      // Undeclared exception!
      try { 
        document0.data();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = new Document("type");
      Element element0 = document0.toggleClass("tv9{/qV,J/Wz[@yB/<");
      boolean boolean0 = element0.hasClass("java.lang.String@0000000009");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$ContainsOwnText");
      document0.toggleClass("org.jsoup.select.Evaluator$ContainsOwnText");
      boolean boolean0 = document0.hasClass("org.jsoup.select.Evaluator$ContainsOwnText");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Document document0 = new Document("type");
      Elements elements0 = document0.getElementsByIndexGreaterThan(1327);
      assertEquals(0, elements0.size());
      
      Element element0 = document0.toggleClass("tv9{/qV,J/Wz[@yB/<");
      boolean boolean0 = element0.hasClass("java.lang.String@0000000009");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$ContainsOwnText");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, " Xonu~ 3vRF>O\"", attributes0);
      String string0 = formElement0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Document document0 = new Document("AfterAfterFrameset");
      Element element0 = document0.val("AfterAfterFrameset");
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("textarea");
      FormElement formElement0 = new FormElement(tag0, ",6.lSuMA", attributes0);
      // Undeclared exception!
      try { 
        formElement0.val((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Document document0 = new Document("yp");
      StringBuilder stringBuilder0 = new StringBuilder("yp");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlHead(stringBuilder0, 2098, document_OutputSettings1);
      assertEquals("yp<#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      Element element0 = document0.createElement("h6");
      Element element1 = element0.appendElement("h6");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "98_tcdiW-TpayFum");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      element1.outerHtmlHead(stringBuilder0, 1, document_OutputSettings1);
      assertEquals("98_tcdiW-TpayFum\n <h6>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      document0.appendElement("br");
      String string0 = document0.html();
      assertEquals("<br>", string0);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Document document0 = new Document("p");
      Element element0 = document0.createElement("p");
      element0.appendElement("p");
      document0.appendChild(element0);
      String string0 = document0.html();
      assertEquals("<p><p></p></p>", string0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Document document0 = new Document("Children collection to be inserted must not be null.");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      // Undeclared exception!
      try { 
        document0.outerHtmlTail((StringBuilder) null, 2508, document_OutputSettings1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      Element element0 = document0.createElement("h6");
      element0.appendElement("h6");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "98_tcdiW-TpayFum");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.outline(true);
      // Undeclared exception!
      element0.outerHtmlTail(stringBuilder0, 10882, document_OutputSettings0);
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      Element element0 = document0.createElement("h6");
      element0.appendElement("h6");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "98_tcdiW-TpayFum");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      element0.appendText("=N(4D.N3");
      document_OutputSettings0.outline(true);
      // Undeclared exception!
      element0.outerHtmlTail(stringBuilder0, 10882, document_OutputSettings0);
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      Document document0 = new Document("sh;o");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.prettyPrint(false);
      document0.outputSettings(document_OutputSettings0);
      String string0 = document0.html();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      Document document0 = new Document("~X");
      boolean boolean0 = document0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Ealuator$ConainsOwnText");
      Document document1 = document0.clone();
      boolean boolean0 = document0.equals(document1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Ealuator$ConainsOwnText");
      Document document1 = document0.clone();
      assertTrue(document1.equals((Object)document0));
      
      document1.prepend("org.jsoup.select.Ealuator$ConainsOwnText");
      boolean boolean0 = document0.equals(document1);
      assertFalse(document1.equals((Object)document0));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test89()  throws Throwable  {
      Document document0 = new Document("98_tcdiW-TpayFum");
      document0.hashCode();
  }
}